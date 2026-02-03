"""
通用工具包模块

提供通用的辅助工具，如时间获取等。

所有工具都是无状态的，直接使用 @tool 装饰器定义。
"""

from datetime import datetime
from langchain_core.tools import tool


@tool
def get_current_time(timezone: str = "local") -> str:
    """
    获取当前的日期和时间

    Args:
        timezone: 时区，默认为本地时区。例如：'UTC', 'Asia/Shanghai'

    Returns:
        当前日期和时间的字符串表示，格式：YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 向后兼容：保留 CommonToolkit 类
class CommonToolkit:
    """
    通用工具包

    .. deprecated::
        直接使用工具函数即可，无需此类。

        使用方式：
        ```python
        from toolkits.common import get_current_time
        ```

    提供不依赖于特定领域的通用工具，如时间获取、系统信息等。
    """

    def get_tools(self):
        """
        返回工具列表

        Returns:
            包含所有通用工具的列表
        """
        return [get_current_time]
