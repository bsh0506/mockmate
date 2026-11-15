"""MockMate 双路助手入口。"""
import asyncio
from hybrid import create_assistant


async def main():
    assistant = await create_assistant()
    print("=" * 60)
    print("  MockMate · 私人面试助手（双路路由版）")
    print("  具体题 → 传统 RAG；关系/全局题 → LightRAG")
    print("=" * 60)
    try:
        while True:
            q = input("\n你问：").strip()
            if q.lower() in ("q", "quit", "exit"):
                print("再见，祝你面试顺利！")
                break
            if not q:
                continue
            print("思考中...")
            try:
                answer = await assistant.answer(q)
            except Exception as e:
                print(f"出错了：{e}")
                continue
            print("\n" + "-" * 60)
            print(answer)
            print("-" * 60)
    except KeyboardInterrupt:
        print("\n\n已退出。")
    finally:
        assistant.close()
        await assistant.rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())