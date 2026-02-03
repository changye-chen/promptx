"""
Web 模块 Schema 定义

定义 Web 工具的输入输出类型。
"""

from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Web 搜索工具输入"""

    query: str = Field(
        description="具体的搜索关键词"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="期望返回的结果条数，范围 1-20"
    )
    categories: str = Field(
        default="general",
        description="搜索类别。可选值: 'general', 'it', 'science', 'news', 'images', 'videos'"
    )
    language: str = Field(
        default="zh-CN",
        description="搜索语言，默认为中文"
    )
    engine: str | None = Field(
        default=None,
        description="指定使用的搜索引擎名称，如 'google', 'bing' 等。默认为 None"
    )


class WebReaderInput(BaseModel):
    """Web 读取工具输入"""

    url: str = Field(
        description="要读取的完整网页 URL"
    )
