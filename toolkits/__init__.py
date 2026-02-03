"""
PromptX 工具包模块

提供可复用的 LangChain 工具集合，用于构建 AI Agent。

## 新的模块结构（v2.0）

```
toolkits/
├── common/          # 通用工具
│   ├── schemas.py
│   └── toolkit.py
├── web/             # Web 工具
│   ├── schemas.py
│   ├── tools.py
│   └── toolkit.py
└── prompt/          # 提示词工程工具
    ├── schemas.py
    ├── templates.py
    ├── tools.py
    └── toolkit.py
```

## 使用方式

### 推荐用法（新代码）
```python
from toolkits.common import CommonToolkit
from toolkits.web import WebToolkit
from toolkits.prompt import PromptToolkit, FileBasedPromptToolkit
```

### 兼容用法（旧代码）
```python
from toolkits import CommonToolkit, WebToolkit, PromptToolkit, FileBasedPromptToolkit
```

两种方式完全兼容，新代码建议使用第一种方式。
"""

# ============================================================================
# 从子模块导入（推荐使用）
# ============================================================================

# Common 工具（可直接导入工具函数）
from toolkits.common import CommonToolkit, get_current_time

# Web 工具（推荐直接导入工具函数）
from toolkits.web import WebToolkit, web_reader, web_search

# Prompt 工具（仍然需要 Toolkit 类）
from toolkits.prompt import FileBasedPromptToolkit, PromptToolkit

# ============================================================================
# 公共 API
# ============================================================================

__all__ = [
    # Common 工具（函数）
    "get_current_time",
    "CommonToolkit",  # 向后兼容
    # Web 工具（函数）
    "web_search",
    "web_reader",
    "WebToolkit",  # 向后兼容
    # Prompt 工具（需要类）
    "PromptToolkit",
    "FileBasedPromptToolkit",
]

# ============================================================================
# 子模块导出（可选）
# ============================================================================

# 也可以直接导入子模块
# from toolkits import common, web, prompt
