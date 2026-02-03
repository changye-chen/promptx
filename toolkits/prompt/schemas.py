"""
工具输入/输出 Schema 定义

使用 Pydantic 定义所有工具的输入和输出类型，提供类型检查和自动文档生成。
"""

from pydantic import BaseModel, Field


class PromptArchitectInput(BaseModel):
    """Prompt Architect 工具输入"""

    requirement: str = Field(
        description="用户的需求描述，支持中文或英文。需要清晰、具体地说明想要实现的功能或目标。"
    )


class DataGeneratorInput(BaseModel):
    """Data Generator 工具输入"""

    num: int = Field(
        default=3,
        ge=1,
        le=100,
        description="生成的测试用例数量，范围 1-100"
    )
    analysis: str = Field(
        description="Prompt Architect 生成的技术规格文档 (JSON 格式)"
    )
    notion: str = Field(
        default="Generate diverse test cases covering edge cases",
        description="特定指令/关注点，例如：'测试边界条件'、'测试多语言支持'"
    )
    require_output: bool = Field(
        default=True,
        description="是否生成预期输出"
    )


class PromptBuilderInput(BaseModel):
    """Prompt Builder 工具输入"""

    analysis: str = Field(
        description="Prompt Architect 生成的技术规格文档 (JSON 格式)"
    )
    test_data: str = Field(
        description="Data Generator 生成的测试数据集 (JSON 格式)"
    )


class PromptEvaluatorInput(BaseModel):
    """Prompt Evaluator 工具输入"""

    analysis: str = Field(
        description="技术规格文档 (JSON 格式)，包含目标和约束"
    )
    input_data: str = Field(
        description="输入给 Agent 的数据"
    )
    actual_output: str = Field(
        description="Agent 实际生成的输出"
    )
    expected_output: str = Field(
        default="",
        description="预期的正确答案（可选）"
    )


class FileBasedToolInput(BaseModel):
    """文件版工具的通用输入参数"""

    num: int = Field(
        default=3,
        ge=1,
        le=100,
        description="生成的测试用例数量，仅 data_generator_file 使用"
    )
    notion: str = Field(
        default="Generate diverse test cases",
        description="特定指令，仅 data_generator_file 使用"
    )
