import json
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, AsrTranscriptionError
from knowledge.processor.import_process.config import get_config

# 模块级缓存：本地 Whisper 模型体积大、加载耗时，整个进程内只加载一次
_ASR_PIPELINE = None


def _get_asr_pipeline(model_path: str, device: str):
    """
    获取(或首次构建)本地Whisper的ASR推理pipeline
    Args:
        model_path: 本地模型路径(HuggingFace格式)
        device:     推理设备(auto/0/-1/ cpu / cuda)

    Returns:
        transformers的自动语音识别pipeline
    """
    global _ASR_PIPELINE
    if _ASR_PIPELINE is None:
        import torch
        from transformers import pipeline as hf_pipeline

        # 设备自动选择：有CUDA用GPU，否则退回CPU
        if device == "auto":
            device = 0 if torch.cuda.is_available() else -1

        # GPU用半精度提速，CPU只支持全精度
        dtype = torch.float16 if (device == 0 or device == "cuda") else torch.float32

        _ASR_PIPELINE = hf_pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device,
            dtype=dtype,
        )
    return _ASR_PIPELINE


class Mp3ToMdNode(BaseNode):
    """
      mp3转换md节点
      位置：导入流程中位于 entry_node 之后（mp3文件专用分支）
      作用：将有声书音频（mp3）通过本地Whisper语音识别模型(whisper-large-v3)转写为文本，
           转写结果以Markdown的形式与源文件同目录保存，后续复用 md_img_node 的处理流程
      模型：WHISPER_PATH（本地HuggingFace格式模型，transformers加载，GPU可用时自动用CUDA）
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

        # 2. 调用本地Whisper模型把音频转写成为文本
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

    def _load_audio_16k(self, import_file_path: Path):
        """
        读取音频文件并统一为Whisper输入格式(16kHz单声道float32)
        Args:
            import_file_path: mp3的文件路径

        Returns:
            (音频数组, 音频时长秒数)
        """
        import numpy as np
        import soundfile as sf

        # 1. 读取音频(libsndfile>=1.1原生支持mp3解码)
        try:
            data, sr = sf.read(str(import_file_path), dtype="float32", always_2d=True)
        except Exception as e:
            raise AsrTranscriptionError(f"读取音频文件失败: {e}", self.name)

        # 2. 多声道混合为单声道
        audio = data.mean(axis=1)

        # 3. 重采样到Whisper要求的16kHz
        if sr != 16000:
            try:
                import torch
                import torchaudio
                wav = torch.from_numpy(audio).unsqueeze(0)
                wav = torchaudio.functional.resample(wav, sr, 16000)
                audio = wav.squeeze(0).numpy().astype(np.float32)
            except Exception as e:
                raise AsrTranscriptionError(f"音频重采样失败(原始采样率{sr}): {e}", self.name)

        return audio, len(audio) / 16000

    def _transcribe_audio_to_text(self, import_file_path: Path) -> str:
        """
        调用本地Whisper语音识别模型将音频转写成为文本
        思路：解码音频为16kHz单声道数组 -> 长音频按30s分块批量送入Whisper -> 拼接转写文本
        Args:
            import_file_path:  mp3的文件路径

        Returns:
            transcript_text: 转写之后的文本
        """
        self.log_step("step2", "读取音频并调用本地Whisper模型转写")

        # 1. 获取配置对象
        config = get_config()

        # 2. 校验模型路径
        model_path = Path(config.whisper_path)
        if not model_path.exists():
            raise AsrTranscriptionError(
                f"本地Whisper模型路径不存在: {config.whisper_path}，请检查.env中的WHISPER_PATH", self.name)

        # 3. 读取并预处理音频
        audio, duration = self._load_audio_16k(import_file_path)
        self.logger.info(f"音频加载完成：{import_file_path.name} 时长:{duration:.1f}s 采样率:16000")

        # 4. 获取ASR pipeline(进程内单例，只在首次调用时加载模型)
        transcribe_start_time = time.time()
        try:
            asr_pipeline = _get_asr_pipeline(config.whisper_path, config.whisper_device)
        except Exception as e:
            raise AsrTranscriptionError(f"加载本地Whisper模型失败: {e}", self.name)

        # 5. 分块转写(长音频按30s切块，5s重叠提升边界处质量)
        try:
            result = asr_pipeline(
                audio,
                chunk_length_s=config.whisper_chunk_length_s,
                stride_length_s=config.whisper_stride_length_s,
                batch_size=config.whisper_batch_size,
                generate_kwargs={
                    "task": "transcribe",
                    "language": config.whisper_language,
                },
            )
            transcript_text = (result.get("text") or "").strip()
        except Exception as e:
            raise AsrTranscriptionError(f"Whisper语音转写失败: {e}", self.name)

        # 6. 校验转写结果非空
        if not transcript_text:
            raise AsrTranscriptionError("Whisper语音转写结果为空", self.name)

        transcribe_end_time = time.time()
        self.logger.info(
            f"Whisper成功转写音频文件：{import_file_path.name} "
            f"音频时长:{duration:.1f}s 转写字数:{len(transcript_text)} 耗时:{transcribe_end_time - transcribe_start_time:.2f}s")

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
