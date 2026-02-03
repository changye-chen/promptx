"""
Prompt 模板管理模块

提供 YAML 模板的加载、渲染和管理功能。
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


class PromptTemplateManager:
    """
    Prompt YAML 模板管理器

    负责从文件系统加载 YAML 格式的 meta prompt 模板，
    并支持变量占位符的渲染。

    Attributes:
        templates_dir: 模板目录路径
    """

    def __init__(self, templates_dir: Path):
        """
        初始化模板管理器

        Args:
            templates_dir: meta prompts YAML 模板目录路径
        """
        self.templates_dir = Path(templates_dir)

    def load_template(self, name: str) -> Dict[str, Any]:
        """
        加载 YAML 格式的 prompt 模板

        Args:
            name: 模板名称（不含 .yaml 后缀）

        Returns:
            包含 messages 列表的字典

        Raises:
            FileNotFoundError: 模板文件不存在
            yaml.YAMLError: YAML 解析失败
        """
        yaml_path = self.templates_dir / f"{name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Template not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def render_messages(self, template: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """
        渲染消息模板，替换变量占位符

        支持的占位符格式：{{variable_name}}

        Args:
            template: 从 load_template() 加载的模板字典
            **kwargs: 模板变量，对应模板中的 {{variable}} 占位符

        Returns:
            渲染后的消息列表，每个消息包含 role 和 content 字段

        Example:
            >>> template = {
            ...     "messages": [
            ...         {"role": "user", "content": "Hello {{name}}!"}
            ...     ]
            ... }
            >>> manager.render_messages(template, name="World")
            [{"role": "user", "content": "Hello World!"}]
        """
        messages = []
        for msg in template["messages"]:
            content = msg["content"]
            # 替换 {{variable}} 格式的占位符
            for key, value in kwargs.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
            messages.append({"role": msg["role"], "content": content})
        return messages


def load_template(name: str, templates_dir: Path = None) -> Dict[str, Any]:
    """
    便捷函数：加载模板

    Args:
        name: 模板名称
        templates_dir: 模板目录，默认为 "meta_prompts"

    Returns:
        模板字典
    """
    manager = PromptTemplateManager(templates_dir or Path("meta_prompts"))
    return manager.load_template(name)


def render_template(template: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
    """
    便捷函数：渲染模板

    Args:
        template: 模板字典
        **kwargs: 模板变量

    Returns:
        渲染后的消息列表
    """
    manager = PromptTemplateManager(Path("."))
    return manager.render_messages(template, **kwargs)
