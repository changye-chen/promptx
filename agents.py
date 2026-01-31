"""
Agent 创建和配置模块

提供预配置的 Agent 实例，使用基于文件系统的提示词生成工作流。
"""

import os
from typing import Any, Iterator, Tuple

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from prompt_toolkit import PromptSession

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
            msg, _ = chunk  # metadata 未使用，用 _ 忽略

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


def create_file_based_prompt_agent(model=None, work_dir: str = "memories") -> CompiledStateGraph:
    """
    创建基于文件系统的提示词生成 Agent

    使用文件 I/O 版本的 prompt 工具包，支持：
    - 中间结果可查看、可编辑
    - 工作流可中断恢复
    - 人工介入调整
    - 对话历史记忆（通过 checkpointer）

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

    # 创建 checkpointer 实现对话记忆
    checkpointer = MemorySaver()

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
        checkpointer=checkpointer,  # 启用对话历史记忆
        system_prompt=f"""你是一个提示词生成专家助手，使用基于文件系统的状态机工作流。

   ## 核心角色
   提示词生成专家助手，使用基于文件系统的状态机工作流。

   ## 工作目录
   /memories/workspace/（持久化到磁盘）

   ## 标准流程
   1. 准备需求 → requirement.txt
   2. 生成规格 → prompt_architect_file() → analysis.json
   3. 生成测试 → data_generator_file() → test_data.json
   4. 生成提示 → prompt_builder_file() → final_prompt.json

   ## 交互原则
   - 先交流理解需求，再执行
   - 按步骤执行，展示中间结果
   - 根据反馈灵活调整
""",
    )

    return agent


if __name__ == "__main__":
    deepseek = get_deepseek_model()
    agent = create_file_based_prompt_agent(model=deepseek)

    # 创建 prompt_toolkit 会话
    session = PromptSession("💬 你: ")

    print("🤖 提示词生成助手已启动！输入 'exit' 或 'quit' 退出\n")

    # 交互式对话循环
    while True:
        try:
            # 获取用户输入（使用 prompt_toolkit，支持中文正确删除）
            user_input = session.prompt().strip()

            # 检查退出命令
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("👋 再见！")
                break

            # 跳过空输入
            if not user_input:
                continue

            # 执行 agent 流式输出
            print("\n🤖 助手: ", end="", flush=True)
            stream = agent.stream(
                input={"messages": [{"role": "user", "content": user_input}]},
                config={"callbacks": [CallbackHandler()], "configurable": {"thread_id": "test_session"}},
                stream_mode=["messages"],
            )

            # 使用美化打印函数
            print_stream(stream)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except EOFError:
            # Ctrl+D 退出
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            continue
