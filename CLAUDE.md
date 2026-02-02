# PromptX 项目上下文文档

> **目标读者**: Claude、ChatGPT 等 AI 助手
> **用途**: 帮助 AI 快速理解项目结构、开发上下文和设计决策

## 项目概述

**PromptX** 是一个系统化的提示词工程智能体助手,通过四步工作流将用户需求转化为高质量、可测试的 AI 提示词。

### 核心价值

- **完整工作流**: 需求 → 技术规格 → 测试数据 → 最终提示词
- **智能体助手**: 通过对话交互完成提示词工程任务
- **状态持久化**: 基于文件系统的状态机,支持中断恢复和人工介入
- **模板化系统**: YAML 格式的 meta prompt 模板,易于扩展
- **中文优先**: 专为中文场景优化,支持中文需求分析

### 项目定位

- **主要形式**: 独立 CLI 应用
- **次要形式**: Python 库(可集成到其他项目)
- **开源策略**: 完全开源
- **目标用户**: AI 应用开发者、提示词工程师、研究者、业务人员

---

## 核心架构

### 四步工作流

```
用户需求 (requirement.txt)
    ↓
1. Prompt Architect → 技术规格 (analysis.json)
    ↓
2. Data Generator → 测试数据 (test_data.json)
    ↓
3. Prompt Builder → 最终提示词 (final_prompt.json)
    ↓
4. Prompt Evaluator → 质量评估
```

### 两种实现模式

#### 1. 内存版 (`PromptToolkit`)
- 数据通过字符串传递
- 适合一次性批量处理
- 无状态,无法中断恢复

#### 2. 文件版 (`FileBasedPromptToolkit`) ⭐
- 数据通过文件系统传递
- **核心特性**:
  - 中间结果可查看、可编辑
  - 工作流可中断恢复
  - 人工介入调整
  - 对话历史记忆(LangGraph Checkpointer)
- **路径系统**:
  - Agent 内部: 相对路径 (如 `requirement.txt`)
  - 实际磁盘: `{work_dir}/workspace/`
  - 临时空间: `/` (内存,会话结束丢失)

### Agent 架构

```
create_deep_agent(deepagents)
├── Model: DeepSeek/OpenAI (LangChain Chat Model)
├── Tools: FileBasedPromptToolkit.get_tools()
├── Store: InMemoryStore (LangGraph)
├── Backend: CompositeBackend
│   ├── default: StateBackend (内存)
│   └── /memories/: FilesystemBackend (磁盘持久化)
└── Checkpointer: MemorySaver (对话历史)
```

---

## 关键组件

### 1. Prompt Toolkit (`toolkits/prompt.py`)

**类**: `PromptToolkit` (内存版) / `FileBasedPromptToolkit` (文件版)

**核心工具**:

| 工具名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| `prompt_architect` | 需求分析 | `requirement.txt` | `analysis.json` |
| `data_generator` | 生成测试数据 | `analysis.json` + `num` | `test_data.json` |
| `prompt_builder` | 构建最终提示词 | `analysis.json` + `test_data.json` | `final_prompt.json` |
| `prompt_evaluator` | 评估提示词质量 | `analysis.json` + 输入输出 | 评分报告 (JSON) |

**数据格式**:
- 所有输出均为 **JSON 格式**
- 支持结构化的 input/output schema
- 包含 task、goal、constraint 等元数据

### 2. Agent 系统 (`agents.py`)

**函数**: `create_file_based_prompt_agent(model, work_dir)`

**系统提示词**:
```
你是一个提示词生成专家助手,使用基于文件系统的状态机工作流。

## 核心角色
提示词生成专家助手,使用基于文件系统的状态机工作流。

## 工作目录
/memories/workspace/ (持久化到磁盘)

## 标准流程
1. 准备需求 → requirement.txt
2. 生成规格 → prompt_architect_file() → analysis.json
3. 生成测试 → data_generator_file() → test_data.json
4. 生成提示 → prompt_builder_file() → final_prompt.json

## 交互原则
- 先交流理解需求,再执行
- 按步骤执行,展示中间结果
- 根据反馈灵活调整
```

**交互界面**:
- 基于 `prompt_toolkit` 的 REPL 循环
- 支持中文输入(正确删除字符)
- 美化的流式输出(区分工具调用和智能体响应)
- 支持 `exit`/`quit`/`退出` 退出

### 3. Meta Prompt 模板 (`meta_prompts/`)

**格式**: YAML 文件

**结构**:
```yaml
messages:
  - role: system
    content: "系统指令..."
  - role: user
    content: "用户指令,包含 {{variable}} 占位符"
```

**模板列表**:
- `prompt_architect.yaml` - Prompt Architect 的 meta prompt
- `data_generator.yaml` - Data Generator 的 meta prompt
- `prompt_builder.yaml` - Prompt Builder 的 meta prompt
- `prompt_evaluator.yaml` - Prompt Evaluator 的 meta prompt

---

## 文件结构

```
promptx/
├── agents.py              # Agent 创建和配置
├── main.py                # 项目入口(空实现,主要在 agents.py)
├── toolkits/
│   ├── __init__.py
│   ├── prompt.py          # Prompt 工具包(核心)
│   ├── common.py          # 通用工具
│   └── web.py             # Web 工具(未使用)
├── meta_prompts/          # Meta Prompt 模板
│   ├── prompt_architect.yaml
│   ├── data_generator.yaml
│   ├── prompt_builder.yaml
│   └── prompt_evaluator.yaml
├── memories/              # 工作空间(持久化存储)
│   └── workspace/         # Agent 工作目录
│       ├── requirement.txt
│       ├── analysis.json
│       ├── test_data.json
│       └── final_prompt.json
├── promptx.ipynb          # Jupyter 实验环境
├── pyproject.toml         # Python 项目配置
├── CLAUDE.md              # 本文档
└── README.md              # 用户文档
```

---

## 技术栈

### 核心依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `deepagents` | ^0.3.8 | Agent 框架(基于 LangGraph) |
| `langchain-deepseek` | ^1.0.1 | DeepSeek 模型集成 |
| `langchain-openai` | ^1.1.7 | OpenAI 模型集成 |
| `langchain-community` | ^0.4.1 | LangChain 社区扩展 |
| `langfuse` | ^3.12.1 | 可观测性和监控 |
| `prompt-toolkit` | ^3.0.52 | 交互式 CLI 界面 |
| `python-dotenv` | ^1.2.1 | 环境变量管理 |
| `pyyaml` | (隐式) | YAML 模板解析 |

### Python 版本
- **要求**: >= 3.12

### 包管理
- **工具**: `uv` (现代 Python 包管理器)
- **配置**: `pyproject.toml` + `uv.lock`

---

## 开发约定

### 代码风格

1. **类型注解**: 函数参数和返回值必须使用类型注解
   ```python
   def _prompt_architect_impl(self, requirement: str) -> str:
   ```

2. **文档字符串**: 使用 Google 风格的 docstring
   ```python
   """
   将用户需求转换为精确的技术规格文档 (JSON)。

   Args:
       requirement (str): 用户的需求描述,支持中文或英文。

   Returns:
       str: JSON 格式的技术规格文档,包含 input/output schema、task、goal、constraint。
   """
   ```

3. **中文注释**: 代码注释使用中文,便于理解

4. **错误处理**: 文件操作需要检查文件是否存在
   ```python
   if not path.exists():
       return f"Error: File '{file_path}' not found"
   ```

### 路径约定

1. **相对路径**: Agent 内部使用相对路径
   ```python
   requirement_path = "requirement.txt"
   ```

2. **绝对路径**: 实际文件操作前拼接工作目录
   ```python
   path = self.work_dir / Path(file_path)
   ```

3. **路径分隔符**: 使用 `pathlib.Path` 而非字符串拼接

### 工具函数约定

1. **内存版方法名**: `_xxx_impl()`
2. **文件版方法名**: `_xxx_file_impl()`
3. **工具注册**: 通过 `get_tools()` 返回 `StructuredTool` 列表

### JSON 格式约定

所有 JSON 输出必须包含:
- `dataset`: 测试用例列表
- `input_schema`: 输入数据结构
- `output_schema`: 输出数据结构
- `task`: 任务描述
- `goal`: 目标说明
- `constraint`: 约束条件

---

## 近期开发规划 (Roadmap)

### 当前阶段: 功能迭代

**优先级排序**:

1. **评估系统** ⭐⭐⭐
   - 完善 `prompt_evaluator` 工具
   - 建立评分标准和评估体系
   - 支持批量评估

2. **自动优化** ⭐⭐⭐
   - 基于评估结果的自动优化循环
   - A/B 测试不同提示词变体
   - 迭代改进 meta prompt

3. **Session Resume** ⭐⭐
   - 对话历史持久化到磁盘(目前只在内存)
   - 支持跨会话恢复对话
   - 保存上下文和工作状态

4. **Rewind 功能** ⭐⭐
   - 回退到上一个请求
   - 撤销所有相关文件改动
   - 保留历史版本

5. **多模型支持** ⭐
   - 集成 `litellm` 库
   - 支持 Claude、GPT-4、本地模型
   - 模型切换和对比

6. **丰富模板库** ⭐
   - 添加常用场景 meta prompt
   - 代码生成、数据分析、写作等
   - 社区贡献模板

---

## 设计决策记录

### 为什么使用文件系统传递数据?

**原因**:
- ✅ 中间结果可查看、可编辑
- ✅ 工作流可中断恢复
- ✅ 人工介入调整
- ✅ 调试友好(可以检查中间文件)
- ❌ 额外的 I/O 开销(可接受)

### 为什么使用 YAML 而非 JSON?

**原因**:
- ✅ 支持注释(可以说明设计意图)
- ✅ 多行字符串更友好
- ✅ 更易读易写
- ❌ 需要 pyyaml 依赖(已接受)

### 为什么使用 deepagents 而非直接用 LangGraph?

**原因**:
- ✅ 封装了 Agent 创建的常用模式
- ✅ 内置 FilesystemBackend(虚拟模式)
- ✅ 简化了 checkpointer 和 store 配置
- ✅ 统一的错误处理

### 为什么当前只用 DeepSeek?

**原因**:
- ✅ 性价比高
- ✅ 中文支持好
- ✅ API 稳定
- 🔄 后续会通过 litellm 支持多模型

---

## 常见任务指南

### 添加新的 Meta Prompt 模板

1. 在 `meta_prompts/` 创建 `xxx.yaml`
2. 定义 messages 列表,使用 `{{variable}}` 占位符
3. 在 `PromptToolkit` 中添加 `_xxx_impl()` 方法
4. 在 `get_tools()` 中注册工具

### 修改 Agent 系统提示词

编辑 `agents.py:139-157` 的 `system_prompt` 参数

### 调试工具调用

1. 运行 `agents.py` 启动交互式 CLI
2. 观察 stderr 的工具调用日志
3. 检查 `memories/workspace/` 中的中间文件
4. 使用 Langfuse 查看 LLM 调用详情

### 运行 Jupyter 实验

```bash
jupyter notebook promptx.ipynb
```

---

## 环境变量配置

在 `.env` 文件中配置:

```bash
# DeepSeek API Key (必需)
DEEP_SEEK_API_KEY=sk-xxx

# OpenAI API Key (可选,用于多模型)
OPENAI_API_KEY=sk-xxx

# Langfuse (可选,用于可观测性)
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 快速开始

### 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -e .
```

### 运行 Agent

```bash
python agents.py
```

### 使用流程

```
你: 我需要一个提示词,用于从文本中提取关键信息

助手: 🔧 调用工具: prompt_architect_file
✅ 工具完成: prompt_architect_file
   输出: ✅ 技术规格已生成: analysis.json

助手: 已生成技术规格,包含 input/output schema...
     是否继续生成测试数据?

你: 继续

助手: 🔧 调用工具: data_generator_file
✅ 工具完成: data_generator_file
   输出: ✅ 测试数据已生成 (3 条): test_data.json

助手: 🔧 调用工具: prompt_builder_file
✅ 工具完成: prompt_builder_file
   输出: ✅ 最终提示词已生成: final_prompt.json

助手: 提示词生成完成!你可以查看 final_prompt.json
```

---

## 注意事项

1. **文件路径**: Agent 内部使用相对路径,工具内部自动映射到 `{work_dir}/workspace/`
2. **中文编码**: 所有文件使用 UTF-8 编码
3. **JSON 格式**: 确保 LLM 输出有效 JSON,可能需要重试或修复
4. **模型限制**: DeepSeek 有速率限制,大量请求可能需要队列
5. **内存管理**: `InMemoryStore` 在会话结束后清空,长期存储需要文件系统

---

## 贡献指南

(待补充)

---

## 许可证

(待确定)

---

**最后更新**: 2026-02-02
**文档版本**: v0.1.0
