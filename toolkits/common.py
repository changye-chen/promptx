"""
通用工具包

提供通用的辅助工具，如时间获取等。
"""

from datetime import datetime
from langchain_core.tools import tool
from typing import List


def _now_tool() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CommonToolkit:
    """通用工具包"""

    def get_tools(self) -> List:
        """返回工具列表"""
        return [tool(_now_tool)]
