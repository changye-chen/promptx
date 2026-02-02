# PromptX 架构设计文档

> **版本**: v0.1.0
> **最后更新**: 2026-02-02
> **读者**: 开发者、架构师、技术贡献者

## 目录

- [1. 系统架构概览](#1-系统架构概览)
- [2. 核心工作流](#2-核心工作流)
- [3. 组件架构](#3-组件架构)
- [4. 文件系统设计](#4-文件系统设计)
- [5. Agent 系统](#5-agent-系统)
- [6. 数据流设计](#6-数据流设计)
- [7. 状态管理](#7-状态管理)
- [8. 扩展性设计](#8-扩展性设计)
- [9. 技术选型](#9-技术选型)

---

## 1. 系统架构概览

### 1.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户交互层                            │
│  ┌──────────────┐         ┌──────────────┐             │
│  │   CLI REPL   │         │  Python API  │             │
│  │ (prompt_toolkit)│       │  (直接调用)   │             │
│  └──────────────┘         └──────────────┘             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Agent 智能体层                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         create_deep_agent (deepagents)           │  │
│  │  - LangGraph StateGraph                          │  │
│  │  - Tool Calling                                  │  │
│  │  - Checkpointer (对话记忆)                        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   工具层 (Tool Layer)                    │
│  ┌──────────────────┬──────────────────┐               │
│  │ PromptToolkit    │ FileBasedPrompt  │               │
│  │ (内存版)         │ Toolkit (文件版) │               │
│  │                  │  ⭐ 生产使用     │               │
│  └──────────────────┴──────────────────┘               │
│         ↓                    ↓                          │
│  ┌──────────────────────────────────────┐             │
│  │   4 个核心工具 (LangChain Tools)      │             │
│  │ - prompt_architect                    │             │
│  │ - data_generator                      │             │
│  │ - prompt_builder                      │             │
│  │ - prompt_evaluator                    │             │
│  └──────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   模板层 (Template Layer)                │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Meta Prompt YAML 模板                    │  │
│  │  - prompt_architect.yaml                         │  │
│  │  - data_generator.yaml                           │  │
│  │  - prompt_builder.yaml                           │  │
│  │  - prompt_evaluator.yaml                         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   持久化层 (Storage Layer)               │
│  ┌──────────────────┬──────────────────┐               │
│  │ FilesystemBackend│   InMemoryStore  │               │
│  │ (/memories/)     │   (LangGraph)    │               │
│  └──────────────────┴──────────────────┘               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   模型层 (Model Layer)                   │
│  ┌──────────────────┬──────────────────┐               │
│  │  ChatDeepSeek    │   (未来) litellm │               │
│  │  langchain_deepseek│  (多模型支持)   │               │
│  └──────────────────┴──────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

1. **可观察性**: 中间结果持久化到文件,可随时检查
2. **可中断性**: 基于文件的状态机,支持中断恢复
3. **可编辑性**: 人工可以介入修改中间结果
4. **可扩展性**: 模块化工具 + YAML 模板,易于添加新能力
5. **简单性**: 避免过度抽象,保持代码直观

---

## 2. 核心工作流

### 2.1 四步提示词工程流程

```
┌──────────────────────────────────────────────────────────┐
│  Step 1: Prompt Architect                                │
│  输入: requirement.txt (用户需求,自然语言)                │
│  输出: analysis.json (技术规格)                          │
│  功能: 理解需求,定义 input/output schema,明确任务目标    │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Step 2: Data Generator                                  │
│  输入: analysis.json + num (测试用例数量)                 │
│  输出: test_data.json (合成测试数据集)                   │
│  功能: 生成多样化测试用例,覆盖边界条件                   │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Step 3: Prompt Builder                                  │
│  输入: analysis.json + test_data.json                    │
│  输出: final_prompt.json (可直接调用的 messages 列表)    │
│  功能: 将规格和数据转化为最终的提示词                     │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Step 4: Prompt Evaluator                                │
│  输入: analysis.json + input_data + actual_output        │
│  输出: evaluation_report.json (评分 + 改进建议)          │
│  功能: 评估提示词质量,提供改进方向                       │
└──────────────────────────────────────────────────────────┘
```

### 2.2 工作流状态机

```
                  ┌─────────────┐
                  │   Initial   │
                  │  (初始状态)  │
                  └──────┬──────┘
                         │
                         ↓
                  ┌─────────────┐
                  │  Require    │
                  │ (收集需求)   │
                  └──────┬──────┘
                         │ prompt_architect_file()
                         ↓
                  ┌─────────────┐
                  │   Analyze   │◄────────┐
                  │ (技术分析)   │         │
                  └──────┬──────┘         │ 人工介入
                         │                │ (修改 analysis.json)
                         ↓ data_generator_file()
                  ┌─────────────┐         │
                  │  Generate   │─────────┘
                  │ (生成数据)   │
                  └──────┬──────┘
                         │ prompt_builder_file()
                         ↓
                  ┌─────────────┐
                  │   Build     │◄────────┐
                  │ (构建提示词) │         │
                  └──────┬──────┘         │ 人工调整
                         │                │ (修改 test_data.json)
                         ↓ prompt_evaluator_file()
                  ┌─────────────┐         │
                  │  Evaluate   │─────────┘
                  │ (评估质量)   │
                  └──────┬──────┘
                         │
                         ↓
                  ┌─────────────┐
                  │  Complete   │
                  │  (完成)     │
                  └─────────────┘
```

**关键特性**:
- 每个状态都有对应的文件输出
- 可以在任意状态中断和恢复
- 人工可以修改文件后继续
- 支持回退到上一个状态

---

## 3. 组件架构

### 3.1 PromptToolkit 类层次

```
PromptToolkit (内存版)
├── __init__(model, meta_prompts_dir)
├── _load_prompt_template(name) -> Dict
├── _render_messages(template, **kwargs) -> List[Dict]
├── _call_llm(messages) -> str
├── _prompt_architect_impl(requirement) -> str
├── _data_generator_impl(num, analysis, notion, require_output) -> str
├── _prompt_builder_impl(analysis, test_data) -> str
├── _prompt_evaluator_impl(analysis, input_data, actual_output, expected_output) -> str
└── get_tools() -> List[StructuredTool]

FileBasedPromptToolkit (文件版)
├── 继承 PromptToolkit 的所有方法
├── __init__(model, work_dir)
├── _read_file(file_path) -> str
├── _write_file(file_path, content) -> None
├── _prompt_architect_file_impl() -> str
├── _data_generator_file_impl(num, notion) -> str
├── _prompt_builder_file_impl() -> str
└── get_tools() -> List[StructuredTool]  # 覆盖父类
```

### 3.2 工具接口规范

所有工具遵循 LangChain `StructuredTool` 接口:

```python
StructuredTool.from_function(
    func=implementation_function,  # 实现函数
    name="tool_name",              # 工具名称(小写下划线)
    description="""工具描述

    Args:
        参数说明

    Returns:
        返回值说明
    """,
)
```

### 3.3 消息格式

**YAML 模板格式**:
```yaml
messages:
  - role: system
    content: |
      你是一个提示词工程师...
  - role: user
    content: |
      用户需求: {{requirement}}

      请生成技术规格...
```

**渲染后的消息格式**:
```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "用户需求: xxx\n\n请生成技术规格..."}
]
```

**LangChain 消息格式**:
```python
[
    SystemMessage(content="..."),
    HumanMessage(content="...")
]
```

---

## 4. 文件系统设计

### 4.1 目录结构

```
promptx/
├── memories/                          # 持久化根目录
│   ├── workspace/                     # Agent 工作空间
│   │   ├── requirement.txt            # 用户需求
│   │   ├── analysis.json              # 技术规格
│   │   ├── test_data.json             # 测试数据集
│   │   ├── final_prompt.json          # 最终提示词
│   │   └── evaluation_report.json     # 评估报告
│   └── checkpoints/                   # 对话历史检查点(未来)
│       └── {thread_id}/
│           └── {checkpoint_id}
├── meta_prompts/                      # Meta Prompt 模板
│   ├── prompt_architect.yaml
│   ├── data_generator.yaml
│   ├── prompt_builder.yaml
│   └── prompt_evaluator.yaml
├── toolkits/                          # 工具包代码
├── agents.py                          # Agent 创建
└── .env                               # 环境变量
```

### 4.2 路径映射规则

| Agent 内部路径 | 实际磁盘路径 | 持久化 |
|---------------|-------------|--------|
| `requirement.txt` | `{work_dir}/workspace/requirement.txt` | ✅ |
| `analysis.json` | `{work_dir}/workspace/analysis.json` | ✅ |
| `test_data.json` | `{work_dir}/workspace/test_data.json` | ✅ |
| `final_prompt.json` | `{work_dir}/workspace/final_prompt.json` | ✅ |
| `/` (临时空间) | 内存 (StateBackend) | ❌ |

**路径映射实现**:
```python
# FileBasedPromptToolkit.__init__
self.work_dir = Path(work_dir) / "workspace"

# _read_file / _write_file
path = self.work_dir / Path(file_path)
```

### 4.3 文件格式规范

**requirement.txt**:
```
我需要一个提示词,用于从新闻文章中提取关键信息:
- 标题
- 作者
- 发布时间
- 关键词
- 摘要
```

**analysis.json**:
```json
{
  "task": "信息提取",
  "goal": "从新闻文章中提取结构化信息",
  "input_schema": {
    "type": "object",
    "properties": {
      "article": {"type": "string", "description": "新闻文章内容"}
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "author": {"type": "string"},
      "publish_time": {"type": "string"},
      "keywords": {"type": "array", "items": {"type": "string"}},
      "summary": {"type": "string"}
    }
  },
  "constraint": "输出必须是有效的 JSON"
}
```

**test_data.json**:
```json
{
  "dataset": [
    {
      "input": {"article": "新闻内容..."},
      "output": {
        "title": "新闻标题",
        "author": "张三",
        "publish_time": "2026-02-02",
        "keywords": ["AI", "技术"],
        "summary": "摘要..."
      }
    }
  ]
}
```

**final_prompt.json**:
```json
[
  {
    "role": "system",
    "content": "你是一个信息提取专家..."
  },
  {
    "role": "user",
    "content": "请从以下文章中提取信息..."
  }
]
```

---

## 5. Agent 系统

### 5.1 Agent 创建流程

```python
agent = create_deep_agent(
    name="file-prompt-agent",
    model=model,                          # ChatDeepSeek
    tools=prompt_toolkit.get_tools(),     # 3个文件工具
    store=InMemoryStore(),                # LangGraph Store
    backend=lambda rt: CompositeBackend(  # 路由后端
        default=StateBackend(rt),         # 默认内存
        routes={
            "/memories/": FilesystemBackend(
                root_dir=real_work_dir,
                virtual_mode=True,        # 虚拟模式(安全)
            ),
        },
    ),
    checkpointer=MemorySaver(),           # 对话历史记忆
    system_prompt="...",                  # 系统提示词
)
```

### 5.2 Agent 状态结构

```python
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "tool", "content": "..."}
    ],
    # 可以添加自定义状态字段
}
```

### 5.3 Backend 路由机制

```
Agent 读写路径
      ↓
CompositeBackend
      ├─ 路径匹配 /memories/?
      │   ├─ 是 → FilesystemBackend (磁盘持久化)
      │   └─ 否 → StateBackend (内存)
      ↓
返回结果给 Agent
```

**virtual_mode=True 的作用**:
- 限制文件操作在 `root_dir` 内
- 防止路径穿越攻击 (如 `../../../etc/passwd`)
- 安全沙箱机制

### 5.4 对话历史管理

```python
# 第一次对话
agent.stream(
    input={"messages": [{"role": "user", "content": "你好"}]},
    config={"configurable": {"thread_id": "test_session"}}
)

# 第二次对话(保留历史)
agent.stream(
    input={"messages": [{"role": "user", "content": "继续"}]},
    config={"configurable": {"thread_id": "test_session"}}  # 相同 thread_id
)
```

**Checkpointer 实现**:
- `MemorySaver()` - 内存存储(当前实现)
- 未来可替换为 `SqliteSaver()` - 持久化到数据库

---

## 6. 数据流设计

### 6.1 完整数据流

```
用户输入 (CLI)
      ↓
HumanMessage
      ↓
LangGraph Agent
      ↓
Tool Call (prompt_architect_file)
      ↓
┌──────────────────────────────────┐
│  FileBasedPromptToolkit          │
│  1. 读取 requirement.txt          │
│  2. 加载 prompt_architect.yaml    │
│  3. 渲染消息(替换 {{variable}})   │
│  4. 调用 DeepSeek 模型            │
│  5. 写入 analysis.json            │
└──────────────────────────────────┘
      ↓
Tool Result (JSON)
      ↓
Agent 推理
      ↓
AIMessage (显示给用户)
      ↓
用户确认继续
      ↓
Tool Call (data_generator_file)
      ↓
... (重复类似流程)
```

### 6.2 模板渲染流程

```python
# 1. 加载 YAML
template = {
    "messages": [
        {"role": "system", "content": "你是{{role}}"},
        {"role": "user", "content": "需求: {{requirement}}"}
    ]
}

# 2. 渲染变量
kwargs = {"role": "提示词工程师", "requirement": "生成一个翻译提示词"}

# 3. 替换占位符
messages = [
    {"role": "system", "content": "你是提示词工程师"},
    {"role": "user", "content": "需求: 生成一个翻译提示词"}
]

# 4. 转换为 LangChain 格式
lc_messages = [
    SystemMessage(content="你是提示词工程师"),
    HumanMessage(content="需求: 生成一个翻译提示词")
]
```

### 6.3 LLM 调用流程

```python
# 1. 构建消息
messages = [SystemMessage(...), HumanMessage(...)]

# 2. 调用模型
response = model.invoke(messages)  # ChatDeepSeek

# 3. 提取内容
content = response.content  # JSON 字符串

# 4. 写入文件
write_file("analysis.json", content)

# 5. 返回结果
return "✅ 技术规格已生成: analysis.json"
```

---

## 7. 状态管理

### 7.1 内存状态 vs 文件状态

| 类型 | 位置 | 生命周期 | 用途 |
|------|------|---------|------|
| **内存状态** | `StateBackend` | 会话结束清除 | Agent 推理上下文 |
| **文件状态** | `FilesystemBackend` | 持久化 | 中间结果、工作数据 |
| **对话历史** | `MemorySaver` | 会话结束清除 | 对话记忆 |

### 7.2 状态转换示例

```python
# 初始状态
state = {"messages": []}

# 用户输入
state["messages"].append({"role": "user", "content": "生成提示词"})

# Agent 调用工具
state["messages"].append({
    "role": "assistant",
    "content": "",
    "tool_calls": [{"name": "prompt_architect_file", "args": {}}]
})

# 工具执行
state["messages"].append({
    "role": "tool",
    "name": "prompt_architect_file",
    "content": "✅ 技术规格已生成: analysis.json"
})

# Agent 响应
state["messages"].append({
    "role": "assistant",
    "content": "已生成技术规格,请查看 analysis.json"
})
```

### 7.3 中断和恢复

**中断场景**:
1. 用户输入 `Ctrl+C`
2. 网络错误
3. LLM API 失败
4. 主动暂停

**恢复流程**:
```python
# 重新启动 Agent
agent = create_file_based_prompt_agent(model)

# 读取之前的工作状态
analysis = read_file("analysis.json")
if analysis:
    # 从 Analyze 状态继续
    print("检测到未完成的工作,从分析阶段继续...")
    # 调用 data_generator_file()
```

---

## 8. 扩展性设计

### 8.1 添加新工具

**步骤**:
1. 创建 YAML 模板 (`meta_prompts/new_tool.yaml`)
2. 在 `PromptToolkit` 添加实现方法
3. 在 `get_tools()` 注册工具

**示例**:
```python
# 1. 添加实现
def _new_tool_impl(self, input_data: str) -> str:
    template = self._load_prompt_template("new_tool")
    messages = self._render_messages(template, data=input_data)
    return self._call_llm(messages)

# 2. 注册工具
def get_tools(self) -> List:
    return [
        # ... 现有工具
        StructuredTool.from_function(
            func=self._new_tool_impl,
            name="new_tool",
            description="新工具描述...",
        ),
    ]
```

### 8.2 添加新模型支持

**当前方案** (硬编码):
```python
from langchain_deepseek import ChatDeepSeek
model = ChatDeepSeek(api_key=api_key)
```

**未来方案** (litellm):
```python
from litellm import completion
response = completion(
    model="deepseek/deepseek-chat",  # 或 "gpt-4", "claude-3"
    messages=messages
)
```

### 8.3 自定义 Meta Prompt

用户可以创建自己的模板目录:

```python
custom_toolkit = FileBasedPromptToolkit(
    model=model,
    work_dir="memories",
    meta_prompts_dir=Path("my_custom_templates")  # 自定义目录
)
```

### 8.4 插件系统 (未来)

```python
# 插件接口
class PromptToolkitPlugin:
    def get_tools(self) -> List[StructuredTool]:
        raise NotImplementedError

    def get_templates(self) -> Dict[str, Path]:
        raise NotImplementedError

# 注册插件
toolkit.register_plugin(MyCustomPlugin())
```

---

## 9. 技术选型

### 9.1 为什么选择这些技术?

| 技术 | 选择理由 | 替代方案 |
|------|---------|---------|
| **LangChain** | 生态成熟,抽象合理 | Langroid, AutoGen |
| **LangGraph** | 状态管理优秀,适合复杂工作流 | LangChain Chain, State Machine |
| **deepagents** | 简化了 Agent 创建,内置 FilesystemBackend | 直接使用 LangGraph |
| **DeepSeek** | 性价比高,中文支持好 | GPT-4, Claude |
| **YAML** | 支持注释,人类可读 | JSON, TOML |
| **prompt_toolkit** | 功能强大,支持中文 | readline, input() |
| **uv** | 速度快,现代依赖管理 | pip, poetry |

### 9.2 架构权衡

| 决策 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| 文件 vs 内存 | 可检查、可编辑 | I/O 开销 | ✅ 文件 |
| 状态机 vs 线性 | 灵活、可中断 | 复杂度 | ✅ 状态机 |
| 模板 vs 硬编码 | 可扩展、可维护 | 学习成本 | ✅ 模板 |
| 单体 vs 微服务 | 简单、易部署 | 难扩展 | ✅ 单体 |

### 9.3 性能考虑

| 操作 | 时间成本 | 优化策略 |
|------|---------|---------|
| 文件读写 | ~10ms | 缓存,异步 |
| LLM 调用 | ~1-5s | 流式输出,批处理 |
| 模板渲染 | ~1ms | 预编译 |
| Agent 推理 | ~100ms | 减少轮次 |

**瓶颈**: LLM 调用时间 >> 其他操作

**结论**: 文件 I/O 开销可忽略,优先考虑可维护性

---

## 10. 安全性设计

### 10.1 FilesystemBackend 安全

```python
FilesystemBackend(
    root_dir=real_work_dir,
    virtual_mode=True  # 启用安全沙箱
)
```

**防护措施**:
- ❌ 阻止路径穿越 (`../../../etc/passwd`)
- ❌ 阻止访问 `root_dir` 外的文件
- ✅ 只允许相对路径操作

### 10.2 API Key 管理

```python
# 使用环境变量,不硬编码
load_dotenv()
api_key = os.getenv("DEEP_SEEK_API_KEY")
```

**最佳实践**:
- ✅ `.env` 文件加入 `.gitignore`
- ✅ 提供 `.env.example` 模板
- ❌ 不在代码中写死 Key

### 10.3 LLM 输入验证

```python
# 验证 JSON 格式
try:
    data = json.loads(llm_output)
except json.JSONDecodeError:
    # 重试或修复 JSON
    return repair_json(llm_output)
```

---

## 11. 监控和调试

### 11.1 Langfuse 集成

```python
from langfuse.langchain import CallbackHandler

langfuse_callback = CallbackHandler()

agent.stream(
    input=...,
    config={"callbacks": [langfuse_callback]}
)
```

**监控指标**:
- LLM 调用次数
- Token 使用量
- 响应时间
- 成本追踪

### 11.2 日志系统

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("生成技术规格...")
logger.debug(f"分析结果: {analysis}")
```

### 11.3 调试技巧

**1. 查看中间文件**:
```bash
cat memories/workspace/analysis.json
```

**2. 启用详细日志**:
```python
logging.basicConfig(level=logging.DEBUG)
```

**3. Langfuse 追踪**:
```bash
# 访问 Langfuse Dashboard
https://cloud.langfuse.com
```

---

## 12. 未来架构演进

### 12.1 短期 (1-3 月)

- [ ] 完善 prompt_evaluator 工具
- [ ] 实现自动测试框架
- [ ] 添加 Session Resume 功能
- [ ] 实现 Rewind 功能

### 12.2 中期 (3-6 月)

- [ ] 集成 litellm 多模型支持
- [ ] 添加 10+ 个 meta prompt 模板
- [ ] Web UI 界面
- [ ] 提示词市场

### 12.3 长期 (6-12 月)

- [ ] 分布式任务队列
- [ ] 多用户支持
- [ ] 团队协作功能
- [ ] SaaS 服务化

---

## 附录

### A. 相关文档

- `CLAUDE.md` - 项目上下文(给 AI)
- `ROADMAP.md` - 发展路线图
- `README.md` - 用户指南

### B. 参考资料

- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [deepagents 仓库](https://github.com/...)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### C. 贡献者

(待补充)

---

**最后更新**: 2026-02-02
**维护者**: @zhonghan
**文档版本**: v0.1.0
