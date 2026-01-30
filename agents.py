"""
Agent 创建和配置模块

提供预配置的 Agent 实例，使用基于文件系统的提示词生成工作流。
"""

import os
from typing import Any, Iterator, Tuple

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langfuse.langchain import CallbackHandler
from langgraph.store.memory import InMemoryStore

from toolkits import FileBasedPromptToolkit

# 加载环境变量
load_dotenv()


def print_stream(stream: Iterator[Tuple[str, Any]]) -> None:
    """
    美化流式输出，区分工具调用和智能体响应

    Args:
        stream: agent.stream() 返回的迭代器
    """
    import sys

    for mode, chunk in stream:
        if mode == "messages":
            msg, metadata = chunk

            # AI 消息（智能体思考过程）
            if isinstance(msg, (AIMessage, AIMessageChunk)):
                # 有工具调用时
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "").strip()  # 获取 name 并去除空白
                        tool_args = tool_call.get("args", {})

                        # 只有当工具名不为空时才显示（过滤流式传输中的空块）
                        if tool_name:
                            print(f"\n🔧 调用工具: {tool_name}", file=sys.stderr)
                            if tool_args:
                                # 格式化参数显示
                                args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
                                print(f"   参数: {args_str}", file=sys.stderr)

                # 有内容时（智能体的回复）
                if hasattr(msg, "content") and msg.content:
                    print(msg.content, end="", flush=True)

            # 工具输出消息
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name
                content = msg.content

                # 简化工具输出显示
                if content and len(content) > 200:
                    preview = content[:200] + "..."
                else:
                    preview = content

                print(f"\n✅ 工具完成: {tool_name}", file=sys.stderr)
                if preview.strip():
                    print(f"   输出: {preview}", file=sys.stderr)

        elif mode == "updates":
            # 状态更新（可选：显示工作进度）
            pass

    print("\n", file=sys.stderr)  # 结束换行


def get_deepseek_model():
    """获取 DeepSeek 模型实例"""
    api_key = os.getenv("DEEP_SEEK_API_KEY")
    return ChatDeepSeek(api_key=api_key, model="deepseek-chat")


def create_file_based_prompt_agent(model=None, work_dir: str = "memories") -> Any:
    """
    创建基于文件系统的提示词生成 Agent

    使用文件 I/O 版本的 prompt 工具包，支持：
    - 中间结果可查看、可编辑
    - 工作流可中断恢复
    - 人工介入调整

    路径系统：
    - Agent 内部：使用相对路径（如 "requirement.txt"）
    - 文件系统后端：/memories/workspace/ -> {work_dir}/workspace/（磁盘持久化）
    - 临时空间：/（内存，会话结束丢失）

    Args:
        model: LangChain 模型，默认使用 DeepSeek
        work_dir: 磁盘持久化根目录（例如 "/home/user/code/promptx/memories"）

    Returns:
        配置好的 deep agent
    """
    if model is None:
        model = get_deepseek_model()

    # 构建真实磁盘路径
    if work_dir.startswith("/"):
        real_work_dir = work_dir
    else:
        real_work_dir = f"/home/zhonghan.chen/code/promptx/{work_dir}"

    prompt_toolkit = FileBasedPromptToolkit(model=model, work_dir=real_work_dir)

    agent = create_deep_agent(
        name="file-prompt-agent",
        model=model,
        tools=prompt_toolkit.get_tools(),
        store=InMemoryStore(),
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memories/": FilesystemBackend(
                    root_dir=real_work_dir,
                    virtual_mode=True,  # 启用虚拟模式，安全限制在目录内
                ),
            },
        ),
        system_prompt=f"""你是一个提示词生成专家，使用基于文件系统的状态机工作流。

## 状态机工作流

工作目录：`/memories/workspace/`（持久化到磁盘）

标准流程（按顺序执行）：

**步骤 1：准备需求**
```
write_file("requirement.txt", "<用户需求>")
```

**步骤 2：生成技术规格**
```
prompt_architect_file()
```
→ 读取 requirement.txt → 写入 analysis.json

**步骤 3：生成测试数据**
```
data_generator_file(num=5)
```
→ 读取 analysis.json → 写入 test_data.json

**步骤 4：生成最终提示词**
```
prompt_builder_file()
```
→ 读取 analysis.json + test_data.json → 写入 final_prompt.json

## 工具说明

### 提示词工程工具（无参数）
- `prompt_architect_file()` - 生成技术规格
- `data_generator_file(num=3)` - 生成测试数据
- `prompt_builder_file()` - 生成最终提示词

### 辅助工具
- `ls`、`read_file`、`write_file`、`edit_file` - 查看和编辑文件
- `web_search`、`web_reader` - 联网搜索和阅读

## 工作原则

1. **全自动执行**：无需与用户交流，直接使用工具完成任务
2. **按顺序调用**：严格按照步骤 1→2→3→4 执行
3. **检查中间结果**：每步完成后可用 `read_file` 查看输出
4. **灵活调整**：如发现问题可用 `edit_file` 修改后继续

## 磁盘映射

`/memories/workspace/` → `{real_work_dir}/workspace/`
""",
    )

    return agent


if __name__ == "__main__":
    deepseek = get_deepseek_model()
    agent = create_file_based_prompt_agent(model=deepseek)

    # 测试流式输出
    user_input = "请帮我生成一个提示词，我将视频通过ASR转换为文本。但是文本中有很多识别错误。请帮我生成一个提示词，能帮我纠正这些ASR错误。并梳理成一个视频内容的总结报告,输出为Markdown格式的文本即可,不需要结构化数据。,输入也就一个文本字符串"
    stream = agent.stream(
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [CallbackHandler()]},
        stream_mode=["messages"],
    )

    # 使用美化打印函数
    print_stream(stream)
