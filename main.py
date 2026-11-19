"""MockMate 双路助手入口：1 问答 / 2 模拟面试。"""
import asyncio
from rag.hybrid import create_assistant
from rag.interview import Interviewer


async def qa_mode(assistant):
    """普通问答模式。"""
    while True:
        q = input("\n你问：").strip()
        if q.lower() in ("q", "quit", "exit"):
            return
        if not q:
            continue
        print("思考中...")
        print("\n" + "-" * 60)
        print(await assistant.answer(q))
        print("-" * 60)


async def interview_mode(assistant):
    """模拟面试模式。"""
    interviewer = Interviewer(assistant)
    direction = input("想面哪个方向？（RAG / Agent / 图谱 / Harness，回车随机）：").strip() or "RAG"

    question, knowledge = interviewer.ask(direction)
    print(f"\n【面试官】{question}")
    messages = [{"role": "assistant", "content": question}]

    for i in range(3):          # 最多追问 3 轮
        ans = input(f"\n【你】第{i+1}轮：").strip()
        if ans.lower() in ("q", "quit", "exit"):
            return
        if not ans:
            continue
        messages.append({"role": "user", "content": ans})
        print("\n思考中...")
        reply = interviewer.chat(knowledge, messages)
        print(f"\n【面试官】{reply}")
        messages.append({"role": "assistant", "content": reply})


async def main():
    assistant = await create_assistant()
    print("=" * 60)
    print("  MockMate · 私人面试助手")
    print("  1 = 问答模式   2 = 模拟面试模式")
    print("=" * 60)
    try:
        while True:
            mode = input("\n选模式（1/2，q 退出）：").strip()
            if mode == "1":
                await qa_mode(assistant)
            elif mode == "2":
                await interview_mode(assistant)
            elif mode.lower() in ("q", "quit", "exit"):
                break
    except KeyboardInterrupt:
        print("\n\n已退出。")
    finally:
        assistant.close()
        await assistant.rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())