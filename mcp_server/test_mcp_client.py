import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path("/extra/caochunhong/gm/pcb_multi_agent")
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "pcb_knowledge_server.py"


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== MCP tools ===")
            for tool in tools.tools:
                print("-", tool.name)

            print("\n=== call pcb_knowledge_search ===")
            result = await session.call_tool(
                "pcb_knowledge_search",
                arguments={
                    "query": "漏孔缺陷会导致什么风险，如何维修？",
                    "defect_type": "漏孔",
                    "top_k": 3,
                },
            )

            # 不同 SDK 版本 content 结构可能略有区别，这里做兼容打印。
            print("raw result type:", type(result))

            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)

            print("\n[DONE] MCP client test finished.")


if __name__ == "__main__":
    asyncio.run(main())
