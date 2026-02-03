"""
Web 工具包模块（已废弃）

保留此文件以向后兼容，建议直接使用工具函数。

新的使用方式：
```python
from toolkits.web import web_search, web_reader
```

旧的使用方式（仍然支持）：
```python
from toolkits.web import WebToolkit
toolkit = WebToolkit()
tools = toolkit.get_tools()
```
"""

import warnings
from typing import List

from .tools import web_reader, web_search


class WebToolkit:
    """
    Web 搜索和网页读取工具包

    .. deprecated::
        直接使用工具函数即可，无需此类。

        使用方式：
        ```python
        from toolkits.web import web_search, web_reader
        ```

    提供互联网搜索和网页内容提取功能，支持 SearXNG 搜索引擎
    和 Crawl4AI 网页爬取服务。

    配置通过环境变量读取：
    - SEARX_URL: SearXNG 搜索引擎 URL
    - CRAWL4AI_URL: Crawl4AI 网页爬取服务 URL
    """

    def __init__(self, searx_url: str = None, crawl4ai_url: str = None):
        """
        初始化 Web 工具包

        .. deprecated::
            参数已废弃，配置从环境变量读取。

        Args:
            searx_url: 已废弃，使用环境变量 SEARX_URL
            crawl4ai_url: 已废弃，使用环境变量 CRAWL4AI_URL
        """
        if searx_url is not None or crawl4ai_url is not None:
            warnings.warn(
                "WebToolkit 的构造参数已废弃，请使用环境变量配置："
                "SEARX_URL 和 CRAWL4AI_URL",
                DeprecationWarning,
                stacklevel=2,
            )

    def get_tools(self) -> List:
        """
        返回工具列表

        Returns:
            包含 web_search 和 web_reader 工具的列表
        """
        return [web_search, web_reader]
