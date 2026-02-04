"""
Agent 创建和配置模块

提供预配置的 Agent 实例，使用基于文件系统的提示词生成工作流。
"""

import os
from typing import Any, Iterator, Tuple

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.protocol import SandboxBackendProtocol, ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox
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
                        tool_name = tool_call.get(
                            "name", ""
                        ).strip()  # 获取 name 并去除空白
                        tool_args = tool_call.get("args", {})

                        # 只有当工具名不为空时才显示（过滤流式传输中的空块）
                        if tool_name:
                            print(f"\n🔧 调用工具: {tool_name}", file=sys.stderr)
                            if tool_args:
                                # 格式化参数显示
                                args_str = ", ".join(
                                    f"{k}={v}" for k, v in tool_args.items()
                                )
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


class HybridSandboxBackend(SandboxBackendProtocol):
    """混合沙盒后端：文件操作直接使用 FilesystemBackend，命令执行使用 subprocess。

    文件操作直接持久化到实际目录，同时支持命令执行。
    """

    def __init__(self, work_dir: str):
        self.filesystem_backend = FilesystemBackend(
            root_dir=work_dir,
            virtual_mode=True,
        )
        self.work_dir = work_dir

    @property
    def id(self) -> str:
        return f"hybrid-sandbox-{self.work_dir}"

    def execute(self, command: str) -> ExecuteResponse:
        """在本地工作目录中执行 shell 命令。

        Args:
            command: 要执行的 shell 命令

        Returns:
            ExecuteResponse 包含输出、退出码等信息
        """
        import subprocess

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr if output else result.stderr

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=False,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output="Error: Command timed out after 120 seconds",
                exit_code=-1,
                truncated=True,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=-1,
                truncated=False,
            )

    def ls_info(self, path: str):
        return self.filesystem_backend.ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        return self.filesystem_backend.read(file_path, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        return await self.filesystem_backend.aread(file_path, offset, limit)

    def write(self, file_path: str, content: str):
        return self.filesystem_backend.write(file_path, content)

    async def awrite(self, file_path: str, content: str):
        return await self.filesystem_backend.awrite(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        return self.filesystem_backend.edit(
            file_path, old_string, new_string, replace_all
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        return await self.filesystem_backend.aedit(
            file_path, old_string, new_string, replace_all
        )

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        return self.filesystem_backend.grep_raw(pattern, path, glob)

    async def agrep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ):
        return await self.filesystem_backend.agrep_raw(pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/"):
        return self.filesystem_backend.glob_info(pattern, path)

    async def aglob_info(self, pattern: str, path: str = "/"):
        return await self.filesystem_backend.aglob_info(pattern, path)

    def upload_files(self, files):
        return self.filesystem_backend.upload_files(files)

    async def aupload_files(self, files):
        return await self.filesystem_backend.aupload_files(files)

    def download_files(self, paths):
        return self.filesystem_backend.download_files(paths)

    async def adownload_files(self, paths):
        return await self.filesystem_backend.adownload_files(paths)


def get_deepseek_model():
    """获取 DeepSeek 模型实例"""
    api_key = os.getenv("DEEP_SEEK_API_KEY")
    return ChatDeepSeek(api_key=api_key, model="deepseek-chat")


def get_openai_model():
    """获取 OpenAI 模型实例"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model)


def create_file_based_prompt_agent(
    model=None, work_dir: str = "memories"
) -> CompiledStateGraph:
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

    # 使用混合后端：文件操作直接持久化，同时支持命令执行
    filesystem_backend = HybridSandboxBackend(work_dir=real_work_dir)

    agent = create_deep_agent(
        name="file-prompt-agent",
        model=model,
        tools=prompt_toolkit.get_tools(),
        store=InMemoryStore(),
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memories/": filesystem_backend,
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
    3. 生成测试数据 → data_generator_file() → test_data.json（固定生成3个示例）
    4. 生成提示 → prompt_builder_file() → final_prompt.json

   ## 文件操作工具使用指南（非常重要）

   你有两个文件写入工具，必须正确选择：

   ### write_file（创建新文件）
   - **用途**：创建全新的文件并写入内容
   - **使用场景**：
     - 文件**不存在**时（如首次生成analysis.json、test_data.json等）
     - 需要**完全覆盖**已有文件时（但这种情况较少见，优先考虑edit_file）
   - **重要**：write_file会**创建或完全覆盖**文件，不适合用于增量修改

   ### edit_file（修改现有文件）
   - **用途**：修改已有文件中的部分内容
   - **使用场景**：
     - 文件**已存在**，需要**更新部分内容**（如修改requirement.txt、调整已有JSON的某个字段）
     - 需要**追加内容**到文件末尾
     - 需要**删除或替换**某些特定内容
   - **重要**：使用edit_file之前**必须先read_file**读取文件内容
   - **参数**：old_string（要替换的旧文本）+ new_string（新文本）

   ### 决策流程
   1. 先使用`ls`或`glob`检查文件是否存在
   2. 如果文件不存在 → 使用`write_file`创建
   3. 如果文件已存在 → 使用`read_file`读取内容 → 使用`edit_file`修改

   ## 命令执行工具（execute）

   你有 `execute` 工具可以在本地环境中执行 shell 命令。

   ### 使用场景
   - 运行测试：`execute(command="pytest tests/")`
   - 构建项目：`execute(command="npm run build")`
   - 检查环境：`execute(command="python3 --version")`
   - 安装依赖：`execute(command="pip install package_name")`
   - 查看系统信息：`execute(command="ls -la /memories/workspace/")`

   ### 重要提示
   - 命令在工作目录 `/memories/workspace/` 中执行
   - 使用绝对路径访问文件（如 `/memories/workspace/file.txt`）
   - 命令有 120 秒超时限制
   - 避免使用 `cd` 命令，直接使用绝对路径

   ## 交互原则
   - 先交流理解需求，再执行
   - 按步骤执行，展示中间结果
   - 根据反馈灵活调整
   - 优先使用edit_file修改现有文件，避免不必要的文件覆盖
""",
    )

    return agent


if __name__ == "__main__":
    deepseek = get_deepseek_model()
    openai = get_openai_model()
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
                config={
                    "callbacks": [CallbackHandler()],
                    "configurable": {"thread_id": "test_session"},
                },
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
