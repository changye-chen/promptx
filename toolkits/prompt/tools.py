"""
独立的工具函数模块

使用 @tool 装饰器定义所有提示词工程工具函数。
这些函数是独立的、可复用的，不依赖任何类状态。
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as langchain_tool

from .templates import PromptTemplateManager


def _call_llm(model, messages: List[Dict[str, str]]) -> str:
    """
    调用 LLM 并返回响应

    Args:
        model: LangChain Chat Model 实例
        messages: 消息列表，每个消息包含 role 和 content

    Returns:
        LLM 响应内容
    """
    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            # 历史对话中的 assistant 消息
            lc_messages.append(AIMessage(content=msg["content"]))

    response = model.invoke(lc_messages)
    return response.content


# ============================================================================
# 内存版本工具（直接传参数）
# ============================================================================


@langchain_tool
def prompt_architect(requirement: str) -> str:
    """
    将用户需求转换为精确的技术规格文档 (JSON)。

    这个工具会分析用户需求，生成包含 input/output schema、task、goal、constraint
    的技术规格文档，为后续的测试数据生成和提示词构建奠定基础。

    Args:
        requirement: 用户的需求描述，支持中文或英文。需要清晰、具体地说明想要实现的功能或目标。

    Returns:
        JSON 格式的技术规格文档，包含以下字段：
        - input_schema: 输入数据结构定义
        - output_schema: 输出数据结构定义
        - task: 任务描述
        - goal: 目标说明
        - constraint: 约束条件

    Example:
        >>> result = prompt_architect.invoke({"requirement": "我需要一个从文本中提取邮箱的工具"})
        >>> # 返回 JSON 格式的技术规格
    """
    # 注意：这个工具需要从外部注入 model 和 template_manager
    # 实际使用时需要通过 Toolkit 类来配置
    raise NotImplementedError(
        "这个工具需要通过 PromptToolkit 类来使用，"
        "因为它依赖于 model 和 template_manager 实例。"
    )


@langchain_tool
def data_generator(
    num: int = 3,
    analysis: str = "",
    notion: str = "Generate diverse test cases covering edge cases",
    require_output: bool = True,
) -> str:
    """
    基于技术规格生成高质量的合成测试数据集。

    这个工具会根据技术规格文档生成多样化的测试用例，覆盖常见场景和边界条件。

    Args:
        num: 生成的测试用例数量，默认 3，范围 1-100
        analysis: Prompt Architect 生成的技术规格 (JSON)
        notion: 特定指令/关注点，例如："测试边界条件" 或 "测试多语言支持"
        require_output: 是否生成预期输出，默认 True

    Returns:
        JSON 格式的数据集，包含：
        - dataset: 测试用例列表，每个用例包含 input 和 output
        - input_schema: 输入数据结构
        - output_schema: 输出数据结构

    Example:
        >>> result = data_generator.invoke({
        ...     "num": 5,
        ...     "analysis": analysis_json,
        ...     "notion": "测试边界条件"
        ... })
    """
    raise NotImplementedError(
        "这个工具需要通过 PromptToolkit 类来使用，"
        "因为它依赖于 model 和 template_manager 实例。"
    )


@langchain_tool
def prompt_builder(analysis: str, test_data: str) -> str:
    """
    将技术规格和测试数据转换为可直接调用的 messages 列表 (JSON)。

    这个工具会综合分析技术规格和测试数据，生成一个完整的、可直接用于
    API 调用的 messages 列表，包含系统提示词和示例对话。

    Args:
        analysis: Prompt Architect 生成的技术规格 (JSON)
        test_data: Data Generator 生成的测试数据集 (JSON)

    Returns:
        JSON 数组，包含完整的 messages 列表，可直接用于 API 调用。
        格式：[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Example:
        >>> result = prompt_builder.invoke({
        ...     "analysis": analysis_json,
        ...     "test_data": test_data_json
        ... })
    """
    raise NotImplementedError(
        "这个工具需要通过 PromptToolkit 类来使用，"
        "因为它依赖于 model 和 template_manager 实例。"
    )


@langchain_tool
def prompt_evaluator(
    analysis: str,
    input_data: str,
    actual_output: str,
    expected_output: str = "",
) -> str:
    """
    评估 AI Agent 的执行结果，返回评分和改进建议。

    这个工具会对比实际输出和预期输出，评估结果质量，并提供改进建议。

    Args:
        analysis: 技术规格文档 (JSON)，包含目标和约束
        input_data: 输入给 Agent 的数据
        actual_output: Agent 实际生成的输出
        expected_output: 预期的正确答案（可选）

    Returns:
        JSON 格式的评估报告，包含：
        - reasoning: 评估理由
        - issues: 发现的问题列表
        - suggestions: 改进建议列表
        - score: 评分 (0-100)

    Example:
        >>> result = prompt_evaluator.invoke({
        ...     "analysis": analysis_json,
        ...     "input_data": input_text,
        ...     "actual_output": actual_result,
        ...     "expected_output": expected_result
        ... })
    """
    raise NotImplementedError(
        "这个工具需要通过 PromptToolkit 类来使用，"
        "因为它依赖于 model 和 template_manager 实例。"
    )


# ============================================================================
# 工具工厂函数
# ============================================================================


def create_prompt_architect_tool(model, template_manager: PromptTemplateManager) -> StructuredTool:
    """创建 prompt_architect 工具实例"""

    def _impl(requirement: str) -> str:
        template = template_manager.load_template("prompt_architect")
        messages = template_manager.render_messages(template, requirement=requirement)
        return _call_llm(model, messages)

    return StructuredTool.from_function(
        func=_impl,
        name="prompt_architect",
        description="""将用户需求转换为精确的技术规格文档 (JSON)。

Args:
    requirement (str): 用户的需求描述，支持中文或英文。

Returns:
    str: JSON 格式的技术规格文档，包含 input/output schema、task、goal、constraint。
""",
    )


def create_data_generator_tool(model, template_manager: PromptTemplateManager) -> StructuredTool:
    """创建 data_generator 工具实例"""

    def _impl(
        num: int = 3,
        analysis: str = "",
        notion: str = "Generate diverse test cases covering edge cases",
        require_output: bool = True,
    ) -> str:
        template = template_manager.load_template("data_generator")
        messages = template_manager.render_messages(
            template,
            num=num,
            analysis=analysis,
            notion=notion,
            require_output=str(require_output).lower(),
        )
        return _call_llm(model, messages)

    return StructuredTool.from_function(
        func=_impl,
        name="data_generator",
        description="""基于技术规格生成高质量的合成测试数据集。

Args:
    num (int): 生成的测试用例数量。
    analysis (str): Prompt Architect 生成的技术规格 (JSON)。
    notion (str): 特定指令/关注点，如 "测试边界条件" 或 "测试多语言支持"。
    require_output (bool): 是否生成预期输出，默认 True。

Returns:
    str: JSON 格式的数据集，包含 dataset 键和测试用例列表。
""",
    )


def create_prompt_builder_tool(model, template_manager: PromptTemplateManager) -> StructuredTool:
    """创建 prompt_builder 工具实例"""

    def _impl(analysis: str, test_data: str) -> str:
        template = template_manager.load_template("prompt_builder")
        messages = template_manager.render_messages(template, analysis=analysis, test_data=test_data)
        return _call_llm(model, messages)

    return StructuredTool.from_function(
        func=_impl,
        name="prompt_builder",
        description="""将技术规格和测试数据转换为可直接调用的 messages 列表 (JSON)。

Args:
    analysis (str): Prompt Architect 生成的技术规格 (JSON)。
    test_data (str): Data Generator 生成的测试数据集 (JSON)。

Returns:
    str: JSON 数组，包含完整的 messages 列表，可直接用于 API 调用。
""",
    )


def create_prompt_evaluator_tool(model, template_manager: PromptTemplateManager) -> StructuredTool:
    """创建 prompt_evaluator 工具实例"""

    def _impl(
        analysis: str,
        input_data: str,
        actual_output: str,
        expected_output: str = "",
    ) -> str:
        template = template_manager.load_template("prompt_evaluator")
        messages = template_manager.render_messages(
            template,
            analysis=analysis,
            input_data=input_data,
            expected_output=expected_output,
            actual_output=actual_output,
        )
        return _call_llm(model, messages)

    return StructuredTool.from_function(
        func=_impl,
        name="prompt_evaluator",
        description="""评估 AI Agent 的执行结果，返回评分和改进建议。

Args:
    analysis (str): 技术规格文档 (JSON)，包含目标和约束。
    input_data (str): 输入给 Agent 的数据。
    actual_output (str): Agent 实际生成的输出。
    expected_output (str): 预期的正确答案（可选）。

Returns:
    str: JSON 格式的评估报告，包含 reasoning、issues、suggestions、score (0-100)。
""",
    )
