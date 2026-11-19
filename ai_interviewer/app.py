"""MockMate 前端：极简对话界面（多会话、流式、思考在上）。"""
import json
import streamlit as st
import requests

API = "http://127.0.0.1:8000"


def sse_events(url, body):
    with requests.post(url, json=body, stream=True) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            yield json.loads(line[5:].strip())


st.set_page_config(page_title="MockMate", page_icon="🎤")

# ---------- 会话状态：问答 / 面试各自独立 ----------
if "qa_sid" not in st.session_state:
    st.session_state.qa_sid = None
if "interview_sid" not in st.session_state:
    st.session_state.interview_sid = None
if "qa_messages" not in st.session_state:
    st.session_state.qa_messages = []
if "interview_messages" not in st.session_state:
    st.session_state.interview_messages = []


def load_session(sid, mode_key):
    """从后端拉取历史会话消息，设为当前。"""
    r = requests.get(f"{API}/api/sessions/{sid}/messages")
    if r.ok:
        msgs = r.json()
        if mode_key == "qa":
            st.session_state.qa_sid = sid
            st.session_state.qa_messages = msgs
        else:
            st.session_state.interview_sid = sid
            st.session_state.interview_messages = msgs


# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("MockMate")
    st.caption("私人面试助手")
    mode = st.radio("模式", ["问答", "模拟面试"], key="mode_radio", label_visibility="collapsed")
    mode_key = "qa" if mode == "问答" else "interview"
    hist_key = f"hist_{mode_key}"   # 每个模式各自的历史选中状态

    # 新建会话
    if mode == "模拟面试":
        direction = st.selectbox("面试方向", ["RAG", "Agent", "图谱", "Harness"])
        if st.button("开始面试", type="primary"):
            r = requests.post(f"{API}/api/interview/start", json={"direction": direction})
            if r.ok:
                data = r.json()
                st.session_state.interview_sid = data["session_id"]
                st.session_state.interview_messages = [{"role": "assistant", "content": data["question"]}]
                st.session_state.pop(hist_key, None)   # 清空历史选中
                st.rerun()
            else:
                st.error(r.json().get("detail", "出错了"))
    else:
        if st.button("新建会话", type="primary"):
            st.session_state.qa_sid = None
            st.session_state.qa_messages = []
            st.session_state.pop(hist_key, None)       # 清空历史选中
            st.rerun()

    # 历史会话（列表样式：标题一行省略号 + 删除按钮）
    st.divider()
    st.caption("历史会话")
    try:
        sessions = requests.get(f"{API}/api/sessions", params={"mode": mode_key}).json()
    except Exception:
        sessions = []

    current_sid = st.session_state.qa_sid if mode_key == "qa" else st.session_state.interview_sid

    for s in sessions:
        title = s["title"] or "未命名会话"
        title = title if len(title) <= 18 else title[:18] + "…"  # 一行省略号，不带时间

        c1, c2 = st.columns([9, 1], vertical_alignment="center")
        with c1:
            if st.button(title, key=f"load_{s['id']}", type="tertiary"):
                load_session(s["id"], mode_key)
                st.rerun()
        with c2:
            if st.button("🗑", key=f"del_{s['id']}", type="tertiary"):
                requests.delete(f"{API}/api/sessions/{s['id']}")
                if current_sid == s["id"]:  # 删的是当前会话 → 清空
                    if mode_key == "qa":
                        st.session_state.qa_sid = None
                        st.session_state.qa_messages = []
                    else:
                        st.session_state.interview_sid = None
                        st.session_state.interview_messages = []
                st.rerun()

# ---------- 当前模式的消息 ----------
msgs = st.session_state.qa_messages if mode_key == "qa" else st.session_state.interview_messages
for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- 底部输入框 ----------
placeholder = "回答面试官..." if mode == "模拟面试" else "问我任何面试问题..."
prompt = st.chat_input(placeholder, submit_mode="disable")

if prompt:
    if mode == "模拟面试" and st.session_state.interview_sid is None:
        st.warning("请先在左侧点「开始面试」")
    else:
        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if mode == "问答":
            with st.chat_message("assistant"):
                thinking_parts = []
                meta = {}
                first_answer = None
                events = sse_events(f"{API}/api/chat/stream",
                                    {"session_id": st.session_state.qa_sid, "question": prompt})

                with st.status("思考中...", expanded=False, type="compact") as status:
                    for event in events:
                        t = event["type"]
                        if t == "thinking":
                            thinking_parts.append(event["delta"])
                        elif t == "answer":
                            first_answer = event["delta"]
                            break
                        elif t == "done":
                            meta.update(event)
                            break
                    if thinking_parts:
                        st.caption("".join(thinking_parts))
                    status.update(label="思考过程", state="complete")

                def stream_answer():
                    if first_answer:
                        yield first_answer
                    for event in events:
                        t = event["type"]
                        if t == "answer":
                            yield event["delta"]
                        elif t == "done":
                            meta.update(event)
                            break

                answer = st.write_stream(stream_answer())

                if meta.get("sources"):
                    with st.expander("引用来源"):
                        for s in meta["sources"]:
                            st.caption(f"[{s['source']}] {s['heading']}")
                st.caption(f"路由：{meta.get('path')} · {meta.get('latency_ms')}ms")

            st.session_state.qa_sid = meta.get("session_id")
            st.session_state.qa_messages.append({"role": "assistant", "content": answer})
        else:
            with st.chat_message("assistant"):
                thinking_parts = []
                first_answer = None
                events = sse_events(f"{API}/api/interview/chat/stream",
                                    {"session_id": st.session_state.interview_sid,
                                     "answer": prompt})

                # 思考：折叠在上方
                with st.status("思考中...", expanded=False, type="compact") as status:
                    for event in events:
                        t = event["type"]
                        if t == "thinking":
                            thinking_parts.append(event["delta"])
                        elif t == "answer":
                            first_answer = event["delta"]
                            break
                        elif t == "done":
                            break
                    if thinking_parts:
                        st.caption("".join(thinking_parts))  # 灰色思考内容
                    status.update(label="思考过程", state="complete")


                # 答案：流式在下方
                def stream_interview():
                    if first_answer:
                        yield first_answer
                    for event in events:
                        if event["type"] == "answer":
                            yield event["delta"]
                        elif event["type"] == "done":
                            break


                reply = st.write_stream(stream_interview())
                st.session_state.interview_messages.append({"role": "assistant", "content": reply})