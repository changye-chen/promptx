# PromptX

> 系统化的提示词工程智能体助手

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

**PromptX** 是一个基于 AI 智能体的提示词工程工具,通过四步工作流将你的需求转化为高质量、可测试的 AI 提示词。

## ✨ 核心特性

- 🔧 **完整工作流** - 需求分析 → 数据生成 → 提示词构建 → 质量评估
- 🤖 **AI 智能体助手** - 通过对话交互完成提示词工程任务
- 💾 **状态持久化** - 基于文件系统的状态机,支持中断恢复和人工介入
- 📝 **模板化系统** - YAML 格式的 meta prompt 模板,易于扩展
- 🌏 **中文优先** - 专为中文场景优化,支持中文需求分析
- 🔍 **可观察性** - 集成 Langfuse,追踪每次 LLM 调用

## 🎯 适用场景

- 📊 **数据提取** - 从文本中提取结构化信息
- ✍️ **内容生成** - 文章写作、营销文案、创意写作
- 🔎 **文本分析** - 情感分析、分类、摘要
- 💻 **代码生成** - 函数、类、算法生成
- 🤝 **问答系统** - FAQ 生成、RAG 提示词设计
- 🌐 **翻译任务** - 多语言翻译和本地化

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/promptx.git
cd promptx

# 安装依赖 (推荐使用 uv)
uv sync

# 或使用 pip
pip install -e .
```

### 配置环境变量

创建 `.env` 文件:

```bash
# DeepSeek API Key (必需)
DEEP_SEEK_API_KEY=sk-your-api-key-here

# Langfuse (可选,用于可观测性)
LANGFUSE_PUBLIC_KEY=pk-your-public-key
LANGFUSE_SECRET_KEY=sk-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 运行

```bash
python agents.py
```

启动后,你将看到交互式对话界面:

```
🤖 提示词生成助手已启动!输入 'exit' 或 'quit' 退出

💬 你: 我需要一个提示词,用于从新闻文章中提取关键信息

🤖 助手: [AI 助手将引导你完成四步工作流]
```

## 📖 使用示例

### 示例 1: 信息提取提示词

```
💬 你: 我需要一个提示词,用于从新闻文章中提取以下信息:
- 标题
- 作者
- 发布时间
- 关键词
- 摘要

🤖 助手: 好的,我来帮你生成这个提示词。

[Step 1] 分析需求...
✅ 技术规格已生成: analysis.json

[Step 2] 生成测试数据...
✅ 测试数据已生成 (3 条): test_data.json

[Step 3] 构建提示词...
✅ 最终提示词已生成: final_prompt.json

🤖 助手: 提示词生成完成!你可以查看 memories/workspace/final_prompt.json

💬 你: 请展示一下

🤖 助手: [显示生成的提示词内容]
```

### 示例 2: 代码生成提示词

```
💬 你: 帮我生成一个 Python 函数提示词,用于计算斐波那契数列

🤖 助手: 好的,我会生成一个专门用于斐波那契数列的代码生成提示词。

[执行工作流...]

✅ 完成!生成的提示词包含:
- 输入参数说明
- 输出格式规范
- 边界条件处理
- 示例测试用例
```

## 🏗️ 工作流说明

PromptX 采用四步工作流:

```
┌──────────────────────────────────────────────────────────┐
│  Step 1: Prompt Architect                                │
│  输入: requirement.txt (用户需求)                         │
│  输出: analysis.json (技术规格)                          │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Step 2: Data Generator                                  │
│  输入: analysis.json + 测试用例数量                      │
│  输出: test_data.json (合成测试数据)                     │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Step 3: Prompt Builder                                  │
│  输入: analysis.json + test_data.json                    │
│  输出: final_prompt.json (最终提示词)                    │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Step 4: Prompt Evaluator                                │
│  输入: analysis.json + 输入输出                          │
│  输出: evaluation_report.json (质量评估)                 │
└──────────────────────────────────────────────────────────┘
```

### 工作文件说明

所有工作文件保存在 `memories/workspace/` 目录:

| 文件 | 说明 |
|------|------|
| `requirement.txt` | 用户需求描述 |
| `analysis.json` | 技术规格文档 (input/output schema、任务目标) |
| `test_data.json` | 测试数据集 |
| `final_prompt.json` | 最终生成的提示词 (可直接用于 API 调用) |
| `evaluation_report.json` | 质量评估报告 |

## 🛠️ 高级功能

### 人工介入调整

你可以在任意步骤中断并修改中间结果:

```bash
# 查看当前工作文件
cat memories/workspace/analysis.json

# 手动编辑
vim memories/workspace/analysis.json

# 继续工作流
# 在 CLI 中输入 "继续"
```

### 查看 Langfuse 追踪

访问 [Langfuse Dashboard](https://cloud.langfuse.com) 查看:
- 每次 LLM 调用的详细日志
- Token 使用量和成本
- 响应时间分析

## 📚 作为 Python 库使用

除了 CLI,你也可以在代码中直接使用:

```python
from agents import create_file_based_prompt_agent
from langchain_deepseek import ChatDeepSeek

# 创建 Agent
model = ChatDeepSeek(api_key="your-api-key")
agent = create_file_based_prompt_agent(
    model=model,
    work_dir="memories"
)

# 使用 Agent
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "帮我生成一个文本摘要提示词"
    }]
})

print(response["messages"][-1]["content"])
```

## 🧪 开发指南

### 添加新的 Meta Prompt 模板

1. 在 `meta_prompts/` 创建 YAML 文件:

```yaml
# meta_prompts/my_custom_tool.yaml
messages:
  - role: system
    content: |
      你是一个专业的{{role}}助手...

  - role: user
    content: |
      用户需求: {{requirement}}

      请生成{{output_type}}...
```

2. 在 `toolkits/prompt.py` 中添加实现:

```python
def _my_custom_tool_impl(self, requirement: str) -> str:
    template = self._load_prompt_template("my_custom_tool")
    messages = self._render_messages(template, requirement=requirement)
    return self._call_llm(messages)
```

3. 在 `get_tools()` 中注册:

```python
StructuredTool.from_function(
    func=self._my_custom_tool_impl,
    name="my_custom_tool",
    description="自定义工具描述...",
)
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_prompt_toolkit.py
```

## 📖 文档

- **[CLAUDE.md](CLAUDE.md)** - 项目上下文 (给 AI 助手)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 架构设计文档
- **[ROADMAP.md](ROADMAP.md)** - 发展路线图
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - 贡献指南 (待创建)

## ❓ 常见问题

### 1. 支持哪些模型?

目前支持 DeepSeek 模型。我们正在集成 litellm,未来将支持:
- Claude (Anthropic)
- GPT-4/GPT-4o (OpenAI)
- 本地模型 (Llama、Qwen via Ollama)

### 2. 如何修改工作目录?

编辑 `agents.py:114-117`:

```python
if work_dir.startswith("/"):
    real_work_dir = work_dir  # 绝对路径
else:
    real_work_dir = f"/your/custom/path/{work_dir}"  # 相对路径
```

### 3. 生成的提示词如何使用?

查看 `final_prompt.json`,它是一个标准的 messages 数组:

```python
import json

# 加载提示词
with open("memories/workspace/final_prompt.json") as f:
    messages = json.load(f)

# 使用 LangChain 调用
from langchain_openai import ChatOpenAI
model = ChatOpenAI()
response = model.invoke(messages)
print(response.content)
```

### 4. 如何提高生成质量?

1. **明确需求** - 在 `requirement.txt` 中详细描述你的需求
2. **调整测试数据** - 编辑 `test_data.json` 添加更多样化的测试用例
3. **人工优化** - 手动编辑中间文件,然后继续工作流
4. **使用评估工具** - 运行 `prompt_evaluator` 获取改进建议

### 5. 数据隐私安全?

- ✅ 所有工作文件存储在本地
- ✅ 只有 LLM 调用会发送数据到 API
- ✅ 支持本地模型 (未来版本)
- ✅ FilesystemBackend 使用安全沙箱模式

## 🔧 配置选项

### 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `DEEP_SEEK_API_KEY` | ✅ | DeepSeek API 密钥 |
| `OPENAI_API_KEY` | ❌ | OpenAI API 密钥 (多模型支持) |
| `LANGFUSE_PUBLIC_KEY` | ❌ | Langfuse 公钥 |
| `LANGFUSE_SECRET_KEY` | ❌ | Langfuse 私钥 |
| `LANGFUSE_HOST` | ❌ | Langfuse 服务器地址 |

### CLI 参数

```bash
python agents.py --help  # (待实现)
```

## 🤝 贡献

欢迎贡献!请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

优先贡献方向:
- 📝 新的 meta prompt 模板
- 🐛 Bug 修复
- 📚 文档改进
- ✅ 测试用例

## 📜 许可证

[MIT License](LICENSE)

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的 LLM 框架
- [deepagents](https://github.com/...) - Agent 创建工具
- [DeepSeek](https://www.deepseek.com/) - 高性价比的 LLM API

## 📮 联系方式

- **项目主页**: [GitHub](https://github.com/your-username/promptx)
- **问题反馈**: [Issues](https://github.com/your-username/promptx/issues)
- **讨论区**: [Discussions](https://github.com/your-username/promptx/discussions)

---

**开始使用 PromptX,让提示词工程变得简单!** 🚀
