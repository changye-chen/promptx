"""
Web 工具函数模块

提供独立的 Web 搜索和网页读取工具函数。

配置通过环境变量读取：
- SEARX_URL: SearXNG 搜索引擎 URL
- CRAWL4AI_URL: Crawl4AI 网页爬取服务 URL
"""

import os
from typing import Literal

import requests
from langchain_core.tools import tool

# 从环境变量读取配置，提供默认值
SEARX_URL = os.getenv("SEARX_URL") or os.getenv("SEARCHX_URL", "https://sousuo.emoe.top/search")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "https://crawl4ai.emoe.top")


@tool
def web_search(
    query: str,
    max_results: int = 5,
    categories: str = "general",
    language: str = "zh-CN",
    engine: str | None = None,
) -> str:
    """
    利用 SearXNG 引擎进行互联网搜索。

    适用于获取实时新闻、技术文档或百科知识。

    Args:
        query: 具体的搜索关键词
        max_results: 期望返回的结果条数，默认为 5
        categories: 搜索类别。可选值: 'general', 'it', 'science', 'news', 'images', 'videos'
        language: 搜索语言，默认为 "zh-CN"
        engine: 指定使用的搜索引擎名称，如 "google", "bing" 等。默认为 None

    Returns:
        格式化的搜索结果列表，每条包含标题、来源链接和内容摘要

    Environment:
        SEARX_URL: SearXNG 搜索引擎 URL（默认：https://sousuo.emoe.top/search）
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
        response = requests.get(SEARX_URL, params=params, timeout=15)
        response.raise_for_status()
        raw_results = response.json().get("results", [])
    except Exception as e:
        return f"搜索失败: {str(e)}"

    # 核心优化：数据清洗
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


@tool
def web_reader(url: str) -> str:
    """
    阅读特定网页的详细内容。

    支持动态加载的网页（如单页应用）。

    Args:
        url: 要读取的完整网页 URL

    Returns:
        网页的正文内容（Markdown 格式）

    Environment:
        CRAWL4AI_URL: Crawl4AI 网页爬取服务 URL（默认：https://crawl4ai.emoe.top）
    """
    payload = {"url": url, "f": "fit"}

    try:
        response = requests.post(CRAWL4AI_URL, json=payload, timeout=30)
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
