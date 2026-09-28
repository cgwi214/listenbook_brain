import json
from pathlib import Path
from typing import Tuple
from bs4 import BeautifulSoup, NavigableString, Tag

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, HtmlConversionError


class HtmlToMdNode(BaseNode):
    """
      html转换md节点
      位置：导入流程中位于 entry_node 之后（html文件专用分支）
      作用：将网页类资料（书籍简介/作者介绍/运营资料等）转换为 Markdown，
           转换结果与源文件同目录，后续复用 md_img_node 的处理流程
    """
    name = "html_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        html文件转换为md文件
        Args:
            state:

        Returns:

        """
        # 1. 对参数校验
        import_file_path, file_dir_path = self._validate_state_inputs_path(state)

        # 2. 读取html的内容并解析成为DOM树
        soup = self._parse_html_to_soup(import_file_path)

        # 3. 将DOM树递归转换为Markdown文本
        md_content = self._convert_soup_to_md(soup)

        # 4. 将转换的结果写入md文件(和源文件同一个目录)
        md_path = self._save_md_file(import_file_path, file_dir_path, md_content)

        # 5. 更新state 字典的md_path
        state['md_path'] = md_path

        # 6. 返回state
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        校验状态的输入路径
        Args:
            state:  该节点接收到的状态

        Returns:
            (html文件的path, 输出目录的path)
        """
        self.log_step("step1", "对状态的路径输入参数做校验")

        # 1. 获取输入html文件路径
        import_file_path = state.get('import_file_path', '')

        # 2. 获取解析后的输出目录
        file_dir = state.get('file_dir', '')

        # 3. 校验输入的文件路径(非空判断)
        if not import_file_path:
            raise ValidationError("转换的文件不存在", self.name)

        # 4. 用Path标准化
        import_file_path_obj = Path(import_file_path)

        # 5. 校验是一个真实的路径
        if not import_file_path_obj.exists():
            raise FileProcessingError("转换的文件路径不存在", self.name)

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

    def _parse_html_to_soup(self, import_file_path: Path) -> BeautifulSoup:
        """
        读取html文件并解析成BeautifulSoup的DOM树
        Args:
            import_file_path:  html的文件路径

        Returns:
            BeautifulSoup对象（DOM树）
        """
        self.log_step("step2", "读取html文件并解析成为DOM树")

        # 1. 读取html的文件内容(部分网页文档不是utf-8编码 用errors做容错)
        try:
            with open(import_file_path, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()
        except IOError as e:
            raise HtmlConversionError(f"读取HTML文件失败: {e}", self.name)

        # 2. 用lxml解析器解析成为DOM树
        soup = BeautifulSoup(html_content, "lxml")

        # 3. 返回DOM树
        return soup

    def _convert_soup_to_md(self, soup: BeautifulSoup) -> str:
        """
        将DOM树转换为Markdown文本
        思路：从body开始 递归遍历每一个标签 按标签语义映射成为Markdown语法
        Args:
            soup:  BeautifulSoup对象

        Returns:
            md_content: 转换后的Markdown文本
        """
        self.log_step("step3", "将DOM树递归转换为Markdown")

        # 1. 转换结果按行收集
        md_lines = []

        # 2. 从body开始遍历(没有body标签就遍历整个文档)
        root = soup.body if soup.body else soup

        # 3. 递归遍历每一个子元素
        for element in root.children:
            self._convert_element(element, md_lines)

        # 4. 合并成为一个完整的Markdown文档
        md_content = "\n".join(line for line in md_lines if line.strip())

        self.logger.info(f"HTML转换成为Markdown完成 共{len(md_content)}个字符")
        return md_content

    def _convert_element(self, element, md_lines: list):
        """
        递归转换单个DOM元素
        Args:
            element:   当前的DOM元素(可能是Tag标签 也可能是NavigableString文本)
            md_lines:  收集转换结果的列表
        """
        # 1. 文本节点(直接收集文本内容)
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                md_lines.append(text)
            return

        # 2. 非标签的其它情况直接忽略
        if not isinstance(element, Tag):
            return

        # 3. 按标签名做语义映射
        tag_name = element.name.lower()

        # 3.1 标题标签 h1~h6 -> # ~ ######
        if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag_name[1])
            md_lines.append("")
            md_lines.append("#" * level + " " + element.get_text(strip=True))
            md_lines.append("")

        # 3.2 段落标签 p -> 普通文本行
        elif tag_name == "p":
            md_lines.append("")
            md_lines.append(element.get_text(" ", strip=True))
            md_lines.append("")

        # 3.3 无序列表 ul -> "- " 列表项
        elif tag_name in ("ul", "ol"):
            md_lines.append("")
            for idx, li in enumerate(element.find_all("li", recursive=False), start=1):
                prefix = "- " if tag_name == "ul" else f"{idx}. "
                md_lines.append(prefix + li.get_text(" ", strip=True))
            md_lines.append("")

        # 3.4 超链接 a -> [文本](地址)
        elif tag_name == "a":
            href = element.get("href", "")
            md_lines.append(f"[{element.get_text(strip=True)}]({href})")

        # 3.5 图片 img -> ![描述](地址)
        elif tag_name == "img":
            src = element.get("src", "")
            alt = element.get("alt", "图片")
            md_lines.append("")
            md_lines.append(f"![{alt}]({src})")
            md_lines.append("")

        # 3.6 引用块 blockquote -> "> "
        elif tag_name == "blockquote":
            md_lines.append("")
            for line in element.get_text("\n", strip=True).split("\n"):
                md_lines.append("> " + line)
            md_lines.append("")

        # 3.7 预格式化代码块 pre -> ``` 代码块 ```
        elif tag_name == "pre":
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(element.get_text())
            md_lines.append("```")
            md_lines.append("")

        # 3.8 行内代码 code -> `代码`
        elif tag_name == "code":
            md_lines.append(f"`{element.get_text(strip=True)}`")

        # 3.9 加粗 strong/b -> **文本**
        elif tag_name in ("strong", "b"):
            md_lines.append(f"**{element.get_text(strip=True)}**")

        # 3.10 斜体 em/i -> *文本*
        elif tag_name in ("em", "i"):
            md_lines.append(f"*{element.get_text(strip=True)}*")

        # 3.11 换行 br -> 空行
        elif tag_name == "br":
            md_lines.append("")

        # 3.12 水平线 hr -> ---
        elif tag_name == "hr":
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

        # 3.13 表格 table -> Markdown表格
        elif tag_name == "table":
            self._convert_table(element, md_lines)

        # 3.14 容器类标签(div/section/article等) -> 递归处理子元素
        else:
            for child in element.children:
                self._convert_element(child, md_lines)

    def _convert_table(self, table_tag: Tag, md_lines: list):
        """
        将table标签转换为Markdown表格
        Args:
            table_tag:  table的DOM标签
            md_lines:   收集转换结果的列表
        """
        # 1. 收集表格的每一行(tr)的每一个单元格(th/td)的文本
        rows = []
        for tr in table_tag.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)

        # 2. 空表格直接返回
        if not rows:
            return

        # 3. 第一行作为表头
        md_lines.append("")
        md_lines.append("| " + " | ".join(rows[0]) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")

        # 4. 剩下的行作为表格的正文
        for row in rows[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("")

    def _save_md_file(self, import_file_path: Path, file_dir_path: Path, md_content: str) -> str:
        """
        将转换好的Markdown内容写入md文件
        Args:
            import_file_path:  html的文件路径
            file_dir_path:     输出的文件目录
            md_content:        转换后的Markdown内容

        Returns:
            md_path: 生成的md文件路径
        """
        self.log_step("step4", "将转换结果写入md文件")

        # 1. 构建md的文件名字(和html同名 只是后缀不一样)
        md_path = file_dir_path / f"{import_file_path.stem}.md"

        # 2. 写入文件
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        except IOError as e:
            raise HtmlConversionError(f"写入Markdown文件失败: {e}", self.name)

        self.logger.info(f"HTML文件已转换为Markdown文件：{md_path}")
        return str(md_path)


###########测试
if __name__ == '__main__':
    setup_logging()
    html_to_md_node = HtmlToMdNode()

    html_to_md_node_init_state = {
        "import_file_path": r"D:\work\shopkeeper_brain-1\knowledge\test\html\哲学.html",
        "file_dir": r"D:\work\shopkeeper_brain-1\knowledge\test\html"
    }
    processed_result = html_to_md_node.process(html_to_md_node_init_state)

    print(json.dumps(processed_result, indent=4, ensure_ascii=False))
