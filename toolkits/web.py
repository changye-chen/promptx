"""
Web 搜索工具包

提供互联网搜索和网页读取功能。
"""

from typing import List

import requests
from langchain_core.tools import StructuredTool


class WebToolkit:
    """Web 搜索和网页读取工具包"""

    def __init__(
        self,
        searx_url: str = "https://sousuo.emoe.top/search",
        crawl4ai_url: str = "https://crawl4ai.emoe.top",
    ):
        """
        初始化 Web 工具包

        Args:
            searx_url: SearXNG 搜索引擎 URL
            crawl4ai_url: Crawl4AI 网页爬取服务 URL
        """
        self.searx_url = searx_url
        self.crawl4ai_url = crawl4ai_url

    def _web_search_impl(
        self,
        query: str,
        max_results: int = 5,
        categories: str = "general",
        language: str = "zh-CN",
        engine: str | None = None,
    ) -> str:
        """
        利用 SearXNG 引擎进行互联网搜索。适用于获取实时新闻、技术文档或百科知识。

        Args:
            query (str): 具体的搜索关键词。
            max_results (int): 期望返回的结果条数，默认为 5。
            categories (str): 搜索类别。可选值: 'general', 'it', 'science', 'news', 'images', 'videos'。
            language (str): 搜索语言。默认为 "zh-CN"。
            engine (str | None): 指定使用的搜索引擎名称，如 "google", "bing" 等。默认为 None，表示使用默认引擎。

        Returns:
            str: 格式化的搜索结果列表，每条包含标题、来源链接和内容摘要。
        """
        params = (
            {
                "q": query,
                "format": "json",
                "engine": engine,
                "categories": categories,
                "language": language,
            }
            if engine
            else {
                "q": query,
                "format": "json",
                "categories": categories,
                "language": language,
            }
        )

        try:
            response = requests.get(self.searx_url, params=params, timeout=15)
            response.raise_for_status()
            raw_results = response.json().get("results", [])
        except Exception as e:
            return f"搜索失败: {str(e)}"

        # --- 核心优化：数据清洗 ---
        processed_results = []
        # 只取前 max_results 条，避免 Token 溢出
        for res in raw_results[:max_results]:
            # 提取 AI 需要的关键信息
            title = res.get("title", "无标题")
            link = res.get("url", "无链接")
            snippet = res.get("content", "无描述")

            # 格式化为易于 AI 阅读的字符串
            processed_results.append(f"标题: {title}\n链接: {link}\n摘要: {snippet}\n---")

        if not processed_results:
            return "未找到相关结果。"

        return "\n".join(processed_results)

    def _web_reader_impl(self, url: str) -> str:
        """
        当你需要阅读特定网页的详细内容时使用此工具。
        支持动态加载的网页（如单页应用）。

        Args:
            url (str): 要读取的完整网页 URL。

        Returns:
            str: 网页的正文内容（Markdown 格式）。
        """
        payload = {"url": url, "f": "fit"}

        try:
            response = requests.post(self.crawl4ai_url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("success") and data.get("markdown"):
                content = data.get("markdown", "")
                if len(content) > 5000:
                    return content[:5000] + "\n\n(内容过长，已自动截断...)"
                return content
            else:
                return f"未能提取内容: {data.get('error', '未知错误')}"

        except Exception as e:
            return f"读取网页失败: {str(e)}"

    def get_tools(self) -> List:
        """返回工具列表"""

        # 创建 web_search 工具
        web_search = StructuredTool.from_function(
            func=self._web_search_impl,
            name="web_search",
            description="""利用 SearXNG 引擎进行互联网搜索。适用于获取实时新闻、技术文档或百科知识。

        Args:
            query (str): 具体的搜索关键词。
            max_results (int): 期望返回的结果条数，默认为 5。
            categories (str): 搜索类别。可选值: 'general', 'it', 'science', 'news', 'images', 'videos'。
            language (str): 搜索语言。默认为 "zh-CN"。
            engine (str | None): 指定使用的搜索引擎名称，如 "google", "bing" 等。默认为 None，表示使用默认引擎。

        Returns:
            str: 格式化的搜索结果列表，每条包含标题、来源链接和内容摘要。
        """,
        )

        # 创建 web_reader 工具
        web_reader = StructuredTool.from_function(
            func=self._web_reader_impl,
            name="web_reader",
            description="""当你需要阅读特定网页的详细内容时使用此工具。
        支持动态加载的网页（如单页应用）。

        Args:
            url (str): 要读取的完整网页 URL。

        Returns:
            str: 网页的正文内容（Markdown 格式）。
        """,
        )

        return [web_search, web_reader]
