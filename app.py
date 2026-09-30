# -*- coding: utf-8 -*-

import json
import threading
from pathlib import Path

import requests
import streamlit as st

import agent
import config
import tools

DEMO_FILE = Path(__file__).resolve().parent / "demo" / "demo_runs.json"

st.set_page_config(page_title="주식 분석 챗봇", page_icon="📈")


def demo_mtime():
    try:
        return DEMO_FILE.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data
def load_demo(mtime):
    """미리 실행해 둔 결과 (없으면 빈 목록). make_demo.py 로 파일이 바뀌면 새로고침만으로 다시 읽음"""
    try:
        return json.loads(DEMO_FILE.read_text(encoding="utf-8")).get("결과", [])
    except (OSError, ValueError):
        return []


@st.cache_resource
def load_agent():
    """실시간 실행 준비. market.db 가 없으면 None"""
    if not tools.DB_PATH.exists():
        return None
    try:
        return agent.Agent()
    except Exception:
        return None


@st.cache_data(ttl=30)
def llm_online():
    """llama-server 가 켜져 있는지 (2초 안에 연결이 안 되면 꺼진 것으로 봄)"""
    try:
        requests.get(f"{config.LM_STUDIO_URL}/models", timeout=2)
        return True
    except requests.RequestException:
        return False


@st.cache_resource
def run_lock():
    return threading.Lock()         # 로컬 LLM은 한 번에 하나씩: 여러 탭에서 물어도 차례로 처리


def md(text):
    """마크다운에서 ~ 와 $ 가 취소선·수식으로 바뀌지 않게"""
    return str(text).replace("~", "\\~").replace("$", "\\$")


def progress(status):
    """챗봇 진행 알림 → 상태 상자에 표시"""
    def on_event(kind, data):
        if kind == "plan":
            lines = [f"의도: {', '.join(data['의도'])}"]
            if data["가정"]:
                lines.append(f"가정: {'; '.join(data['가정'])}")
            lines.append("계획: " + (" → ".join(s["도구"] for s in data["단계"]) or "자료 모으기 없음"))
            status.write(md("  \n".join(lines)))
        elif kind == "step_start":
            no, step = data
            status.update(label=f"자료 모으는 중 ({no}) {step['도구']}: {step['소질문'][:40]}")
        elif kind == "step":
            if data["오류"]:
                tail = data["결과"][4:90]
            else:
                tail = data["메모"]["결론"][:90] if data.get("메모") else "결과 받음"
            status.write(md(f"{'✗' if data['오류'] else '✓'} {data['번호']}. {data['도구']} — {tail}"))
        elif kind == "check_start":
            status.update(label="빠진 자료가 없는지 점검하는 중…")
        elif kind == "check" and data["추가 단계"]:
            status.write(md("추가로 확인: " + ", ".join(s["도구"] for s in data["추가 단계"])))
        elif kind == "answer_start":
            status.update(label="답 쓰는 중…")
    return on_event


def show_evidence(res, saved=False):
    """답 아래 '근거 보기': 계획 → 단계별 메모와 원본"""
    plan = res["계획"]
    if not res["단계"]:
        return
    with st.expander("근거 보기 (계획과 단계별 자료)"):
        head = [f"**질문 정리** {plan['질문 정리']}", f"**의도** {', '.join(plan['의도'])}"]
        if plan["가정"]:
            head.append(f"**가정** {'; '.join(plan['가정'])}")
        st.markdown(md("  \n".join(head)))
        if plan["답에 담을 것"]:
            st.markdown("**답에 담을 것**  \n" + "  \n".join(md(f"{i}. {x}") for i, x in enumerate(plan["답에 담을 것"], 1)))
        for s in res["단계"]:
            st.markdown(f"**{md(s['번호'])}. {md(s['소질문'])}**  \n`{s['도구']}` `{json.dumps(s['인자'], ensure_ascii=False)}`")
            if s["오류"]:
                st.error(s["결과"])
                continue
            n = s.get("메모")
            if n:
                lines = [f"결론: {n['결론']}"]
                if n["핵심 숫자"]:
                    lines.append("핵심 숫자: " + "; ".join(n["핵심 숫자"]))
                if n["근거"]:
                    lines.append("근거: " + "; ".join(n["근거"]))
                if n["빈 곳"]:
                    lines.append("빈 곳: " + n["빈 곳"])
                st.markdown(md("  \n".join(lines)))
            try:
                st.json(json.loads(s["결과"]), expanded=False)
            except (ValueError, TypeError):
                st.code(s["결과"][:3000])
        m = res.get("요약", {})
        st.caption(f"{'미리 실행한 결과 · ' if saved else ''}LLM {m.get('LLM 호출', '?')}회 · "
                   f"{m.get('걸린 시간(초)', '?')}초")


def show(res, saved=False):
    with st.chat_message("user"):
        st.markdown(md(res["질문"]))
    with st.chat_message("assistant"):
        st.markdown(md(res["답"]))
        show_evidence(res, saved)


# ───────────────────────── 화면 ─────────────────────────
st.title("주식 분석 챗봇")
demos = load_demo(demo_mtime())
bot = load_agent()
live = bot is not None and llm_online()
st.caption("질문마다 따로 답합니다. 앞 대화를 기억하지 않으니 종목과 기간을 넣어 물어봐 주세요."
           + ("" if live else "  \n지금은 실시간 실행이 꺼져 있어 미리 실행해 둔 결과만 볼 수 있습니다."))

if "history" not in st.session_state:
    st.session_state.history = []       # 이번에 새로 물은 질문·답 (화면 표시용, 챗봇에는 넘기지 않음)

with st.sidebar:
    if demos:
        st.subheader("미리 실행한 질문")
        st.markdown("  \n".join(f"{i}. {md(d['질문'])}" for i, d in enumerate(demos, 1)))
    if st.session_state.history:
        st.divider()
        if st.button("새로 물은 내용 지우기", width="stretch"):
            st.session_state.history = []
            st.rerun()

if not demos and not live:
    st.info("보여 줄 결과가 없습니다. 내 PC에서 질문을 실행한 뒤 make_demo.py 로 demo/demo_runs.json 을 만드세요.")
for res in demos:                       # 첫 화면: 미리 실행해 둔 결과
    show(res, saved=True)
if demos and st.session_state.history:
    st.divider()
for res in st.session_state.history:
    show(res)

hint = f"예: {demos[0]['질문']}" if demos else "예: 종목 이름과 기간을 넣어 궁금한 점"
question = st.chat_input(hint if live else "지금은 실시간 실행이 꺼져 있습니다", disabled=not live)
if question and live:
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        status = st.status("질문 이해하고 계획 세우는 중…", expanded=True)
        try:
            with run_lock():
                res = bot.run(question, on_event=progress(status))
        except Exception as e:           # LLM 서버 연결 끊김 등
            status.update(label="실패", state="error", expanded=True)
            st.error(str(e))
            st.stop()
        m = res["요약"]
        status.update(label=f"완료 · {m['걸린 시간(초)']}초 · LLM {m['LLM 호출']}회", state="complete", expanded=False)
        st.markdown(md(res["답"]))
        show_evidence(res)
    res.pop("폴더", None)
    st.session_state.history.append(res)
    st.rerun()                          # 옆 막대의 '지우기' 버튼까지 다시 그림
