import json
import time
import base64
from pathlib import Path
from typing import Tuple

from processor.import_process.base import BaseNode, setup_logging
from processor.import_process.state import ImportGraphState
from processor.import_process.exceptions import ValidationError, FileProcessingError, AsrTranscriptionError
from processor.import_process.config import get_config


class Mp3ToMdNode(BaseNode):
    """
      mp3转换md节点
      位置：导入流程中位于 entry_node 之后（mp3文件专用分支）
      作用：将有声书音频（mp3）通过在线ASR语音识别模型(qwen3-asr-flash)转写为文本，
           转写结果以Markdown的形式与源文件同目录保存，后续复用 md_img_node 的处理流程
      模型：ASR_MODEL（DashScope公共兼容端点，音频以base64经chat.completions的input_audio上传）
    """
    name = "mp3_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        mp3文件转写为md文件
        Args:
            state:

        Returns:

        """
        # 1. 对参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2. 调用在线ASR模型把音频转写成为文本
        transcript_text = self._transcribe_audio_to_text(import_file_path)

        # 3. 将转写文本写入md文件(和源文件同一个目录)
        md_path = self._save_transcript_to_md(import_file_path, file_dir_path, transcript_text)

        # 4. 更新state 字典的md_path
        state['md_path'] = md_path

        # 5. 返回state
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        校验状态的输入路径
        Args:
            state:  该节点接收到的状态

        Returns:
            (mp3文件的path, 输出目录的path)
        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入mp3文件路径
        import_file_path = state.get('import_file_path', '')

        # 2. 获取解析后的输出目录
        file_dir = state.get('file_dir', '')

        # 3. 校验输入的文件路径(非空判断)
        if not import_file_path:
            raise ValidationError("转写的文件不存在", self.name)

        # 4. 用Path标准化
        import_file_path_obj = Path(import_file_path)

        # 5. 校验是一个真实的路径
        if not import_file_path_obj.exists():
            raise FileProcessingError("转写的文件路径不存在", self.name)

        # 6. 判断输出目录是否为空
        if not file_dir:
            # 默认目录做兜底
            file_dir = import_file_path_obj.parent

        # 7. 标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 8. 返回 输入文件以及输出目录的标准path
        return import_file_path_obj, file_dir_path_obj

    def _transcribe_audio_to_text(self, import_file_path: Path) -> str:
        """
        调用在线ASR语音识别模型将音频转写成为文本
        思路：校验大小 -> 读取音频并base64编码 -> 经chat.completions的input_audio多模态输入上传 -> 提取转写文本
        Args:
            import_file_path:  mp3的文件路径

        Returns:
            transcript_text: 转写之后的文本
        """
        self.log_step("step2", "读取音频并调用在线ASR模型转写")

        # 1. 获取配置对象
        config = get_config()

        # 2. 前置校验音频大小(云端多模态接口对文件大小有限制，超限直接拒绝并给出明确提示)
        file_size = import_file_path.stat().st_size
        if file_size > config.asr_max_audio_size:
            raise AsrTranscriptionError(
                f"音频文件过大：{import_file_path.name}({file_size / 1024 / 1024:.1f}MB)，"
                f"超过ASR接口上限({config.asr_max_audio_size / 1024 / 1024:.0f}MB)，"
                f"请压缩或截取后再导入", self.name)

        # 3. 读取音频文件并编码为base64
        try:
            audio_bytes = import_file_path.read_bytes()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except IOError as e:
            raise AsrTranscriptionError(f"读取音频文件失败: {e}", self.name)

        # 4. 构建OpenAI客户端(DashScope公共兼容端点)
        from openai import OpenAI
        client = OpenAI(
            api_key=config.asr_api_key,
            base_url=config.asr_api_base,
        )

        # 5. 调用ASR模型转写(音频以data URI形式经input_audio多模态输入上传)
        transcribe_start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=config.asr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": f"data:audio/mpeg;base64,{audio_b64}",
                                    "format": "mp3",
                                },
                            }
                        ],
                    }
                ],
            )
            transcript_text = (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise AsrTranscriptionError(f"ASR语音转写失败: {e}", self.name)

        # 6. 校验转写结果非空
        if not transcript_text:
            raise AsrTranscriptionError("ASR语音转写结果为空", self.name)

        transcribe_end_time = time.time()
        self.logger.info(
            f"ASR成功转写音频文件：{import_file_path.name} "
            f"文件大小:{file_size / 1024 / 1024:.2f}MB 转写字数:{len(transcript_text)} "
            f"耗时:{transcribe_end_time - transcribe_start_time:.2f}s")

        # 7. 返回转写的文本
        return transcript_text

    def _save_transcript_to_md(self, import_file_path: Path, file_dir_path: Path, transcript_text: str) -> str:
        """
        将转写的文本写入md文件
        Args:
            import_file_path:  mp3的文件路径
            file_dir_path:     输出的文件目录
            transcript_text:   转写之后的文本

        Returns:
            md_path: 生成的md文件路径
        """
        self.log_step("step3", "将转写文本写入md文件")

        # 1. 构建md的文件名字(和mp3同名 只是后缀不一样)
        md_path = file_dir_path / f"{import_file_path.stem}.md"

        # 2. 写入文件(标题 + 转写正文)
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# {import_file_path.stem}\n\n")
                f.write(transcript_text)
        except IOError as e:
            raise AsrTranscriptionError(f"写入Markdown文件失败: {e}", self.name)

        self.logger.info(f"MP3文件已转写为Markdown文件：{md_path}")
        return str(md_path)


###########测试
if __name__ == '__main__':
    setup_logging()
    mp3_to_md_node = Mp3ToMdNode()

    mp3_to_md_node_init_state = {
        "import_file_path": r"D:\work\shopkeeper_brain-1\knowledge\test\mp3\0001.mp3",
        "file_dir": r"D:\work\shopkeeper_brain-1\knowledge\test\mp3"
    }
    processed_result = mp3_to_md_node.process(mp3_to_md_node_init_state)

    print(json.dumps(processed_result, indent=4, ensure_ascii=False))
