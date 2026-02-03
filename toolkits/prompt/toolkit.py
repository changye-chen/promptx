"""
Prompt 工具包模块

提供提示词工程工作流工具，支持内存版本和文件 I/O 版本。

重构说明：
- 使用 PromptTemplateManager 管理模板
- 使用工厂函数创建工具实例
- Toolkit 类只负责 get_tools() 方法
- 文件版不再继承内存版，而是独立实现
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.tools import StructuredTool

from .templates import PromptTemplateManager
from .tools import (
    create_data_generator_tool,
    create_prompt_architect_tool,
    create_prompt_builder_tool,
)


class PromptToolkit:
    """
    提示词工程工具包（内存版本）

    通过参数传递数据的原始版本，适合一次性批量处理。

    Attributes:
        model: LangChain Chat Model (如 ChatDeepSeek, ChatOpenAI)
        template_manager: 模板管理器实例
    """

    def __init__(
        self,
        model,
        meta_prompts_dir: Optional[Path] = None,
    ):
        """
        初始化 Prompt 工具包

        Args:
            model: LangChain Chat Model (如 ChatDeepSeek, ChatOpenAI)
            meta_prompts_dir: meta prompts YAML 模板目录路径，默认为 "meta_prompts"
        """
        self.model = model
        self.template_manager = PromptTemplateManager(
            meta_prompts_dir or Path("meta_prompts")
        )

    def get_tools(self) -> List[StructuredTool]:
        """
        返回工具列表（内存版本）

        Returns:
            StructuredTool 列表，包含 prompt_architect, data_generator, prompt_builder
        """
        return [
            create_prompt_architect_tool(self.model, self.template_manager),
            create_data_generator_tool(self.model, self.template_manager),
            create_prompt_builder_tool(self.model, self.template_manager),
        ]


class FileBasedPromptToolkit:
    """
    提示词工程工具包（文件 I/O 版本）

    通过文件系统传递数据，实现可编辑、可检查的工作流。

    路径说明：
    - Agent 使用相对路径（如 "requirement.txt", "analysis.json"）
    - 工具内部自动映射到真实磁盘路径（{work_dir}/workspace/）
    - 所有文件操作相对于工作目录

    Attributes:
        model: LangChain Chat Model
        work_dir: 真实磁盘工作目录（例如："/home/user/code/promptx/memories/workspace"）
        template_manager: 模板管理器实例
    """

    def __init__(
        self,
        model,
        work_dir: str,
        meta_prompts_dir: Optional[Path] = None,
    ):
        """
        初始化文件版 Prompt 工具包

        Args:
            model: LangChain Chat Model
            work_dir: 真实磁盘工作目录的根路径（不包括 workspace 子目录）
                      例如："/home/user/code/promptx/memories"
                      实际文件将保存在：{work_dir}/workspace/
            meta_prompts_dir: meta prompts YAML 模板目录路径，默认为 "meta_prompts"
        """
        self.model = model
        # 真实磁盘路径
        self.work_dir = Path(work_dir) / "workspace"
        self.template_manager = PromptTemplateManager(
            meta_prompts_dir or Path("meta_prompts")
        )

    def _read_file(self, file_path: str) -> str:
        """
        读取文件内容

        Args:
            file_path: 相对于 work_dir 的文件路径

        Returns:
            文件内容字符串，如果文件不存在则返回错误消息
        """
        path = self.work_dir / Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(self, file_path: str, content: str) -> None:
        """
        写入文件内容

        Args:
            file_path: 相对于 work_dir 的文件路径
            content: 要写入的内容
        """
        path = self.work_dir / Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _prompt_architect_file_impl(self) -> str:
        """
        [文件版] 将用户需求转换为技术规格文档。

        固定流程：
        1. 读取 /memories/workspace/requirement.txt
        2. 生成技术规格 JSON
        3. 写入 /memories/workspace/analysis.json

        Returns:
            操作结果消息
        """
        # 固定路径
        requirement_path = "requirement.txt"
        output_path = "analysis.json"

        # 读取需求
        requirement = self._read_file(requirement_path)
        if requirement.startswith("Error:"):
            return f"❌ 错误：找不到需求文件 {requirement_path}"

        # 调用 LLM 生成
        template = self.template_manager.load_template("prompt_architect")
        from .tools import _call_llm

        messages = self.template_manager.render_messages(template, requirement=requirement)
        result = _call_llm(self.model, messages)

        # 写入文件
        self._write_file(output_path, result)
        return f"✅ 技术规格已生成: {output_path}"

    def _data_generator_file_impl(
        self,
        num: int = 3,
        notion: str = "Generate diverse test cases",
    ) -> str:
        """
        [文件版] 基于技术规格生成测试数据。

        固定流程：
        1. 读取 /memories/workspace/analysis.json
        2. 生成测试数据集（num 条）
        3. 写入 /memories/workspace/test_data.json

        Args:
            num: 生成测试用例数量，默认 3
            notion: 特定指令，如 "测试边界条件" 或 "测试多语言支持"

        Returns:
            操作结果消息
        """
        # 固定路径
        analysis_path = "analysis.json"
        output_path = "test_data.json"

        # 读取分析
        analysis = self._read_file(analysis_path)
        if analysis.startswith("Error:"):
            return f"❌ 错误：找不到技术规格文件 {analysis_path}"

        # 调用 LLM 生成
        template = self.template_manager.load_template("data_generator")
        from .tools import _call_llm

        messages = self.template_manager.render_messages(
            template,
            num=num,
            analysis=analysis,
            notion=notion,
            require_output="true",
        )
        result = _call_llm(self.model, messages)

        # 写入文件
        self._write_file(output_path, result)
        return f"✅ 测试数据已生成 ({num} 条): {output_path}"

    def _prompt_builder_file_impl(self) -> str:
        """
        [文件版] 生成最终提示词。

        固定流程：
        1. 读取 /memories/workspace/analysis.json
        2. 读取 /memories/workspace/test_data.json
        3. 生成最终 messages 列表
        4. 写入 /memories/workspace/final_prompt.json

        Returns:
            操作结果消息
        """
        # 固定路径
        analysis_path = "analysis.json"
        test_data_path = "test_data.json"
        output_path = "final_prompt.json"

        # 读取文件
        analysis = self._read_file(analysis_path)
        if analysis.startswith("Error:"):
            return f"❌ 错误：找不到技术规格文件 {analysis_path}"

        test_data = self._read_file(test_data_path)
        if test_data.startswith("Error:"):
            return f"❌ 错误：找不到测试数据文件 {test_data_path}"

        # 调用 LLM 生成
        template = self.template_manager.load_template("prompt_builder")
        from .tools import _call_llm

        messages = self.template_manager.render_messages(template, analysis=analysis, test_data=test_data)
        result = _call_llm(self.model, messages)

        # 写入文件
        self._write_file(output_path, result)
        return f"✅ 最终提示词已生成: {output_path}"

    def get_tools(self) -> List[StructuredTool]:
        """
        返回工具列表（文件版本）

        Returns:
            StructuredTool 列表，包含 prompt_architect_file, data_generator_file, prompt_builder_file
        """
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(
                func=self._prompt_architect_file_impl,
                name="prompt_architect_file",
                description="""将用户需求转换为技术规格文档。

固定流程：
- 读取：/memories/workspace/requirement.txt
- 写入：/memories/workspace/analysis.json

使用场景：完成需求收集后，第一步调用此工具。
""",
            ),
            StructuredTool.from_function(
                func=self._data_generator_file_impl,
                name="data_generator_file",
                description="""基于技术规格生成测试数据。

固定流程：
- 读取：/memories/workspace/analysis.json
- 写入：/memories/workspace/test_data.json

参数：
- num (int): 生成测试用例数量，默认 3
- notion (str): 特定指令，如 "测试边界条件"

使用场景：完成技术规格后，第二步调用此工具。
""",
            ),
            StructuredTool.from_function(
                func=self._prompt_builder_file_impl,
                name="prompt_builder_file",
                description="""生成最终提示词。

固定流程：
- 读取：/memories/workspace/analysis.json 和 test_data.json
- 写入：/memories/workspace/final_prompt.json

使用场景：完成测试数据后，最后一步调用此工具。
""",
            ),
        ]
