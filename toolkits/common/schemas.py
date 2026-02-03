"""
Common 模块 Schema 定义

定义通用工具的输入输出类型。
"""

from pydantic import BaseModel, Field


class GetCurrentTimeInput(BaseModel):
    """获取当前时间工具输入"""

    timezone: str = Field(
        default="local",
        description="时区，默认为本地时区。例如：'UTC', 'Asia/Shanghai'"
    )
