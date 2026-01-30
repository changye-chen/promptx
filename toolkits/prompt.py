"""
Prompt 工具包

提供提示词工程工作流工具，支持内存版本和文件 I/O 版本。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool


class PromptToolkit:
    """
    提示词工程工具包（内存版本）

    通过大字符串传递数据的原始版本。
    """

    def __init__(self, model=None, meta_prompts_dir: Optional[Path] = None):
        """
        初始化 Prompt 工具包

        Args:
            model: LangChain Chat Model (如 ChatDeepSeek, ChatOpenAI)
            meta_prompts_dir: meta prompts YAML 模板目录路径
        """
        self.model = model
        self.meta_prompts_dir = meta_prompts_dir or Path("meta_prompts")

    def _load_prompt_template(self, name: str) -> Dict[str, Any]:
        """加载 YAML 格式的 prompt 模板"""
        yaml_path = self.meta_prompts_dir / f"{name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _render_messages(self, template: Dict, **kwargs) -> List[Dict[str, str]]:
        """渲染消息模板，替换变量占位符"""
        messages = []
        for msg in template["messages"]:
            content = msg["content"]
            # 替换 {{variable}} 格式的占位符
            for key, value in kwargs.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
            messages.append({"role": msg["role"], "content": content})
        return messages

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用 LLM 并返回响应"""
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # 历史对话中的 assistant 消息转换为 HumanMessage
                lc_messages.append(HumanMessage(content=f"(Previous assistant response): {msg['content']}"))

        response = self.model.invoke(lc_messages)
        return response.content

    def _prompt_architect_impl(self, requirement: str) -> str:
        """
        将用户需求转换为精确的技术规格文档 (JSON)。

        Args:
            requirement (str): 用户的需求描述，支持中文或英文。

        Returns:
            str: JSON 格式的技术规格文档，包含 input/output schema、task、goal、constraint。
        """
        template = self._load_prompt_template("prompt_architect")
        messages = self._render_messages(template, requirement=requirement)
        return self._call_llm(messages)

    def _data_generator_impl(
        self,
        num: int,
        analysis: str,
        notion: str = "Generate diverse test cases covering edge cases",
        require_output: bool = True,
    ) -> str:
        """
        基于技术规格生成高质量的合成测试数据集。

        Args:
            num (int): 生成的测试用例数量。
            analysis (str): Prompt Architect 生成的技术规格 (JSON)。
            notion (str): 特定指令/关注点，如 "测试边界条件" 或 "测试多语言支持"。
            require_output (bool): 是否生成预期输出，默认 True。

        Returns:
            str: JSON 格式的数据集，包含 dataset 键和测试用例列表。
        """
        template = self._load_prompt_template("data_generator")
        messages = self._render_messages(
            template,
            num=num,
            analysis=analysis,
            notion=notion,
            require_output=str(require_output).lower(),
        )
        return self._call_llm(messages)

    def _prompt_builder_impl(self, analysis: str, test_data: str) -> str:
        """
        将技术规格和测试数据转换为可直接调用的 messages 列表 (JSON)。

        Args:
            analysis (str): Prompt Architect 生成的技术规格 (JSON)。
            test_data (str): Data Generator 生成的测试数据集 (JSON)。

        Returns:
            str: JSON 数组，包含完整的 messages 列表，可直接用于 API 调用。
        """
        template = self._load_prompt_template("prompt_builder")
        messages = self._render_messages(template, analysis=analysis, test_data=test_data)
        return self._call_llm(messages)

    def _prompt_evaluator_impl(
        self,
        analysis: str,
        input_data: str,
        actual_output: str,
        expected_output: str = "None provided",
    ) -> str:
        """
        评估 AI Agent 的执行结果，返回评分和改进建议。

        Args:
            analysis (str): 技术规格文档 (JSON)，包含目标和约束。
            input_data (str): 输入给 Agent 的数据。
            actual_output (str): Agent 实际生成的输出。
            expected_output (str): 预期的正确答案（可选）。

        Returns:
            str: JSON 格式的评估报告，包含 reasoning、issues、suggestions、score (0-100)。
        """
        template = self._load_prompt_template("prompt_evaluator")
        messages = self._render_messages(
            template,
            analysis=analysis,
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
        )
        return self._call_llm(messages)

    def get_tools(self) -> List:
        """返回工具列表"""
        return [
            StructuredTool.from_function(
                func=self._prompt_architect_impl,
                name="prompt_architect",
                description="""将用户需求转换为精确的技术规格文档 (JSON)。

        Args:
            requirement (str): 用户的需求描述，支持中文或英文。

        Returns:
            str: JSON 格式的技术规格文档，包含 input/output schema、task、goal、constraint。
        """,
            ),
            StructuredTool.from_function(
                func=self._data_generator_impl,
                name="data_generator",
                description="""基于技术规格生成高质量的合成测试数据集。

        Args:
            num (int): 生成的测试用例数量。
            analysis (str): Prompt Architect 生成的技术规格 (JSON)。
            notion (str): 特定指令/关注点。
            require_output (bool): 是否生成预期输出，默认 True。

        Returns:
            str: JSON 格式的数据集，包含 dataset 键和测试用例列表。
        """,
            ),
            StructuredTool.from_function(
                func=self._prompt_builder_impl,
                name="prompt_builder",
                description="""将技术规格和测试数据转换为可直接调用的 messages 列表 (JSON)。

        Args:
            analysis (str): Prompt Architect 生成的技术规格 (JSON)。
            test_data (str): Data Generator 生成的测试数据集 (JSON)。

        Returns:
            str: JSON 数组，包含完整的 messages 列表，可直接用于 API 调用。
        """,
            ),
        ]


class FileBasedPromptToolkit(PromptToolkit):
    """
    提示词工程工具包（文件 I/O 版本）

    通过文件系统传递数据，实现可编辑、可检查的工作流。

    路径说明：
    - 模型使用相对路径（如 "requirement.txt", "analysis.json"）
    - 工具内部自动映射到真实磁盘路径（{work_dir}/workspace/）
    - 所有文件操作相对于工作目录 /memories/workspace/
    """

    def __init__(self, model, work_dir: str, **kwargs):
        """
        初始化文件版 Prompt 工具包

        Args:
            model: LangChain Chat Model
            work_dir: 真实磁盘工作目录的根路径（不包括 workspace 子目录）
                      例如："/home/user/code/promptx/memories"
                      实际文件将保存在：{work_dir}/workspace/
            **kwargs: 传递给父类的其他参数
        """
        super().__init__(model, **kwargs)
        # 真实磁盘路径
        self.work_dir = Path(work_dir) / "workspace"

    def _read_file(self, file_path: str) -> str:
        """读取文件内容"""
        path = self.work_dir / Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_file(self, file_path: str, content: str) -> None:
        """写入文件内容"""
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
            str: 操作结果消息
        """
        # 固定路径
        requirement_path = "requirement.txt"
        output_path = "analysis.json"

        # 读取需求
        requirement = self._read_file(requirement_path)
        if requirement.startswith("Error:"):
            return f"❌ 错误：找不到需求文件 {requirement_path}"

        # 调用 LLM 生成
        template = self._load_prompt_template("prompt_architect")
        messages = self._render_messages(template, requirement=requirement)
        result = self._call_llm(messages)

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
            num (int): 生成测试用例数量，默认 3
            notion (str): 特定指令，如 "测试边界条件" 或 "测试多语言支持"

        Returns:
            str: 操作结果消息
        """
        # 固定路径
        analysis_path = "analysis.json"
        output_path = "test_data.json"

        # 读取分析
        analysis = self._read_file(analysis_path)
        if analysis.startswith("Error:"):
            return f"❌ 错误：找不到技术规格文件 {analysis_path}"

        # 调用 LLM 生成
        template = self._load_prompt_template("data_generator")
        messages = self._render_messages(
            template,
            num=num,
            analysis=analysis,
            notion=notion,
            require_output="true",
        )
        result = self._call_llm(messages)

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
            str: 操作结果消息
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
        template = self._load_prompt_template("prompt_builder")
        messages = self._render_messages(template, analysis=analysis, test_data=test_data)
        result = self._call_llm(messages)

        # 写入文件
        self._write_file(output_path, result)
        return f"✅ 最终提示词已生成: {output_path}"

    def get_tools(self) -> List:
        """返回工具列表（文件版本）"""
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
