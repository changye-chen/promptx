"""
Web 模块 - Web 工具包

提供互联网搜索和网页读取功能。

配置通过环境变量：
- SEARX_URL: SearXNG 搜索引擎 URL
- CRAWL4AI_URL: Crawl4AI 网页爬取服务 URL
"""

from .tools import web_reader, web_search
from .toolkit import WebToolkit

__all__ = [
    "web_search",
    "web_reader",
    "WebToolkit",  # 向后兼容
]
