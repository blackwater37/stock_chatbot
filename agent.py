# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import time
import traceback

import agent_prompts as P
import config
import llm
import tools
from agent_store import RunStore
from tools import TOOLS, ToolError

MAX_STEPS = 5               # 계획 단계 수 상한
MAX_EXTRA = 2               # 점검에서 추가할 수 있는 단계 수
NOTE_MIN = 700              # 결과가 이보다 짧으면 메모 없이 원본을 그대로 결합에 넘김
MAX_RESULT_CHARS = 6000     # 메모를 쓸 때 LLM에 넘길 결과 최대 길이
#  구조 출력(계획·인자·메모·점검): 반복 억제(repeat/presence/DRY)는 JSON 을 깎으므로 끈다
P_STRUCT = dict(temperature=0.1, top_k=20, top_p=0.95, min_p=0.05,
                repeat_penalty=1.0, presence_penalty=0, dry_multiplier=0)
#  최종 답(문장)
P_ANSWER = dict(temperature=0.4, top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.0, presence_penalty=0)


# ───────────────────────── JSON 형식 (LLM 출력 강제) ─────────────────────────
S = {"type": "string"}


def arr(items, n):
    return {"type": "array", "items": items, "maxItems": n}


def obj(props, required=None):
    return {"type": "object", "properties": props,
            "required": list(props) if required is None else required, "additionalProperties": False}


def arg_props(name):
    return {a: ({"type": "string", "enum": s["choices"]} if s["choices"]
                else {"type": "integer"} if s["kind"] == "int" else S) for a, s in TOOLS[name]["args"].items()}


def step_schema():
    """단계 하나: 도구마다 쓸 수 있는 인자 이름을 강제 (값은 비워 둘 수 있음 → 나중에 채움)"""
    return {"anyOf": [obj({"소질문": S, "도구": {"const": n}, "인자": obj(arg_props(n), required=[]),
                           "앞결과필요": {"type": "boolean"}}) for n in TOOLS]}


def plan_schema():
    return obj({"질문 정리": S, "의도": arr({"type": "string", "enum": P.INTENTS}, 3), "종목": arr(S, 5), "기간": S,
                "가정": arr(S, 3), "되물을 것": S, "답에 담을 것": arr(S, 5), "단계": arr(step_schema(), MAX_STEPS)})


def args_schema(name):
    need = [a for a, s in TOOLS[name]["args"].items() if s["required"]]
    return obj({"인자": obj(arg_props(name), required=need)})


NOTE_SCHEMA = obj({"결론": S, "핵심 숫자": arr(S, 8), "근거": arr(S, 5), "빈 곳": S})


def check_schema():
    return obj({"빠진 것": arr(S, 5), "추가 단계": arr(step_schema(), MAX_EXTRA)})


# ───────────────────────── 답 검사 ─────────────────────────
def clip(s, n):
    return re.sub(r"\s+", " ", str(s or "")).strip()[:n]


def strs(x, n, k=200):
    return [clip(v, k) for v in (x if isinstance(x, list) else []) if clip(v, k)][:n]


def clean_step(s):
    if not isinstance(s, dict) or s.get("도구") not in TOOLS:
        return None
    allowed = TOOLS[s["도구"]]["args"]
    args = {k: v for k, v in (s.get("인자") if isinstance(s.get("인자"), dict) else {}).items()
            if k in allowed and v not in (None, "")}
    return {"소질문": clip(s.get("소질문"), 200) or s["도구"], "도구": s["도구"], "인자": args,
            "앞결과필요": bool(s.get("앞결과필요"))}


def step_key(name, args):
    return json.dumps([name, args], ensure_ascii=False, sort_keys=True)


def v_plan(d):
    if not isinstance(d, dict) or not isinstance(d.get("단계"), list):
        raise ValueError("단계 없음")
    steps, seen = [], set()
    for s in d["단계"]:
        c = clean_step(s)
        if c and step_key(c["도구"], c["인자"]) not in seen:
            seen.add(step_key(c["도구"], c["인자"]))
            steps.append(c)
    return {"질문 정리": clip(d.get("질문 정리"), 300), "의도": [x for x in strs(d.get("의도"), 3) if x in P.INTENTS] or ["기타"],
            "종목": strs(d.get("종목"), 5, 40), "기간": clip(d.get("기간"), 60), "가정": strs(d.get("가정"), 3),
            "되물을 것": clip(d.get("되물을 것"), 200), "답에 담을 것": strs(d.get("답에 담을 것"), 5),
            "단계": steps[:MAX_STEPS]}


def v_args(name):
    def check(d):
        a = d.get("인자") if isinstance(d, dict) else None
        if not isinstance(a, dict):
            raise ValueError("인자 없음")
        return {k: v for k, v in a.items() if k in TOOLS[name]["args"] and v not in (None, "")}
    return check


def v_note(d):
    if not isinstance(d, dict) or not clip(d.get("결론"), 10):
        raise ValueError("결론 없음")
    return {"결론": clip(d["결론"], 400), "핵심 숫자": strs(d.get("핵심 숫자"), 8, 80),
            "근거": strs(d.get("근거"), 5, 150), "빈 곳": clip(d.get("빈 곳"), 200)}


def v_check(d):
    if not isinstance(d, dict):
        raise ValueError("형식 오류")
    return {"빠진 것": strs(d.get("빠진 것"), 5),
            "추가 단계": [c for c in map(clean_step, d.get("추가 단계") or []) if c][:MAX_EXTRA]}


def parse_json(text):
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e < s:
        raise ValueError("JSON 없음")
    return json.loads(t[s:e + 1])


# ───────────────────────── 숫자 확인 ─────────────────────────
NUM = re.compile(r"(?<![\w.])([-+]?\d[\d,]*(?:\.\d+)?)\s*(조|억|만)?")
UNIT = {"조": 1e12, "억": 1e8, "만": 1e4}


def _numbers(text):
    """(보이는 글자, 값, 허용 오차, %인지). 오차 = 적힌 자릿수의 반 (12% → 0.5, 12.1% → 0.05)"""
    out = []
    for m in NUM.finditer(text or ""):
        raw = m.group(1).replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        mul = UNIT.get(m.group(2), 1)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        pct = text[m.end():m.end() + 1] == "%"
        out.append((m.group(0).strip() + ("%" if pct else ""), abs(v) * mul, 0.5 * 10 ** -dec * mul, pct))
    return out


def unverified(text, sources):
    """text 속 숫자 중 sources 어디에도 없는 것 (날짜·연도·작은 정수는 넘어감)"""
    pool = [v for s in sources for _, v, _, _ in _numbers(s)]
    bad = []
    for tok, v, tol, pct in _numbers(text):
        if not pct and v == int(v) and (v <= 31 or 1990 <= v <= 2100):
            continue
        if not any(abs(v - p) <= max(tol, 0.006 * p) + 1e-9 for p in pool):
            bad.append(tok)
    return list(dict.fromkeys(bad))[:8]


def note_text(n):
    return " ".join([n.get("결론", ""), *n.get("핵심 숫자", []), *n.get("근거", [])])


# ───────────────────────── LLM 호출 ─────────────────────────
class LLM:
    """질문 하나 동안의 LLM 호출 (횟수·시간 기록). 서버가 JSON 형식 강제를 거부하면 끄고 계속"""
    schema_on = True

    def __init__(self):
        self.calls, self.seconds = 0, 0.0

    def _chat(self, system, user, max_tokens, **params):
        t0 = time.time()
        try:
            return llm.chat(system, user, max_tokens=max_tokens, return_truncated=True, **params)
        finally:
            self.calls += 1
            self.seconds += time.time() - t0

    def json(self, kind, system, user, schema, validate, max_tokens=1200):
        system = f"{system}\n\n출력: JSON 객체 하나. 키 이름은 그대로: {', '.join(schema['properties'])}"
        for _ in range(3):
            params = dict(P_STRUCT)
            if LLM.schema_on:
                params["response_format"] = {"type": "json_schema", "json_schema": {"name": kind, "schema": schema}}
            try:
                text, truncated = self._chat(system, user, max_tokens, **params)
            except RuntimeError as e:
                if "연결 불가" in str(e):
                    raise
                if LLM.schema_on and re.search(r"grammar|schema|response_format", str(e), re.I):
                    LLM.schema_on = False
                    print("  (서버가 JSON 형식 강제를 받지 않아 끄고 계속합니다)")
                continue
            try:
                return validate(parse_json(text))
            except Exception:
                if truncated:
                    max_tokens *= 2
        return None

    def text(self, system, user, max_tokens=1500):
        text, _ = self._chat(system, user, max_tokens, **P_ANSWER)
        return text.strip()


# ───────────────────────── 도구 실행 ─────────────────────────
def run_tool(name, args, verbose=False):
    t = TOOLS[name]
    missing = [a for a, s in t["args"].items() if s["required"] and args.get(a) in (None, "")]
    if missing:
        return f"오류: 필요한 인자가 없습니다: {', '.join(missing)}"
    try:
        out = t["fn"](**args)
    except ToolError as e:
        return f"오류: {e}"
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return f"오류: 도구 실행 중 문제 ({type(e).__name__}: {e})"
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)


def tool_text(brief=False):
    lines = []
    for name, t in TOOLS.items():
        sig = ", ".join(f"{a}{'' if s['required'] else '(선택)'}" for a, s in t["args"].items())
        if brief:
            lines.append(f"- {name}({sig}): {t['desc'][:70]}")
            continue
        lines.append(f"- {name}({sig}): {t['desc']}")
        lines += [f"    {a}: {s['desc']}" for a, s in t["args"].items()]
        lines.append(f"    예: {json.dumps({'도구': name, '인자': t['example']}, ensure_ascii=False)}")
    return "\n".join(lines)


def notes_text(steps, with_raw=True):
    """단계 메모를 LLM에게 보여 줄 글로"""
    out = []
    for s in steps:
        out.append(f"({s['번호']}) {s['소질문']} — {s['도구']} {json.dumps(s['인자'], ensure_ascii=False)}")
        n = s.get("메모")
        if s["오류"]:
            out.append(f"  오류: {s['결과'][:300]}")
        elif n:
            out.append(f"  결론: {n['결론']}")
            if n["핵심 숫자"]:
                out.append(f"  핵심 숫자: {'; '.join(n['핵심 숫자'])}")
            if n["근거"]:
                out.append(f"  근거: {'; '.join(n['근거'])}")
            if n["빈 곳"]:
                out.append(f"  빈 곳: {n['빈 곳']}")
        elif with_raw:
            out.append(f"  결과: {s['결과'][:NOTE_MIN * 2]}")
    return "\n".join(out) or "(단계 없음)"


# ───────────────────────── 에이전트 ─────────────────────────
class Agent:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.store = RunStore()
        self.cache = {}             # (도구, 인자) → (결과, 메모): 질문 하나 안에서만 재사용, 끝나면 비움
        self.emit = lambda kind, data: None     # 진행 알림 (화면이 받아서 표시)
        last, catalog, full, brief = tools.last_day(), tools.data_catalog(), tool_text(), tool_text(brief=True)
        self.catalog = catalog
        self.sys_plan = P.PLAN_SYS.format(last_day=last, catalog=catalog, tools=full, examples=P.PLAN_EXAMPLES)
        self.sys_args = P.ARGS_SYS.format(tools=full)
        self.sys_check = P.CHECK_SYS.format(catalog=catalog, tools=brief)
        self.sys_answer = P.ANSWER_SYS.format(last_day=last, catalog=catalog)

    def say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # 1) 계획
    def plan(self, question, L):
        plan = L.json("plan", self.sys_plan, f"질문: {question}", plan_schema(), v_plan, 1800)
        if plan is None:        # 계획 실패: 질문 그대로 움직임·뉴스를 볼 수 없으니 되묻기
            plan = {"질문 정리": question, "의도": ["기타"], "종목": [], "기간": "", "가정": [],
                    "되물을 것": "질문을 이해하지 못했습니다. 종목과 기간을 넣어 다시 물어봐 주세요.", "답에 담을 것": [], "단계": []}
        return plan

    # 2) 단계 실행
    def execute(self, no, step, done, run, L):
        self.emit("step_start", (no, step))
        name, args = step["도구"], dict(step["인자"])
        spec = TOOLS[name]["args"]
        missing = [a for a, s in spec.items() if s["required"] and not args.get(a)]
        if step["앞결과필요"] or missing:
            args.update(self.fill_args(step, args, done, L) or {})
        t0 = time.time()
        key = step_key(name, args)
        reused = key in self.cache
        if reused:
            result, note, bad = self.cache[key]
        else:
            result = run_tool(name, args, self.verbose)
            if result.startswith("오류:"):             # 오류 내용을 보여 주고 인자를 한 번 고치게 함
                fixed = self.fill_args(step, args, done, L, error=result)
                if fixed and fixed != args:
                    args = {**args, **fixed}
                    key = step_key(name, args)
                    result = run_tool(name, args, self.verbose)
            note, bad = (None, [])
            if not result.startswith("오류:") and len(result) > NOTE_MIN:
                note, bad = self.note(step["소질문"], name, args, result, L)
            self.cache[key] = (result, note, bad)
        rec = {"번호": no, "소질문": step["소질문"], "도구": name, "인자": args, "결과": result,
               "오류": result.startswith("오류:"), "메모": note, "확인 안 된 숫자": bad, "재사용": reused,
               "걸린 시간(초)": round(time.time() - t0, 1)}
        run.save_step(no, rec)
        self.emit("step", rec)
        self.say(f"  [단계 {no}] {name} {json.dumps(args, ensure_ascii=False)} → "
                 + ("오류: " + result[3:80] if rec["오류"] else f"결과 {len(result):,}자"
                    + (f", 메모: {note['결론'][:60]}…" if note else ", 메모 없이 원본 사용")))
        return rec

    def fill_args(self, step, args, done, L, error=None):
        user = (f"[소질문] {step['소질문']}\n[도구] {step['도구']}\n[지금 인자] {json.dumps(args, ensure_ascii=False)}\n"
                + (f"[오류] {error}\n" if error else "") + f"[앞 단계 메모]\n{notes_text(done)}")
        return L.json("args", self.sys_args, user, args_schema(step["도구"]), v_args(step["도구"]), 400)

    def note(self, subq, name, args, result, L):
        user = f"[소질문] {subq}\n[도구] {name} {json.dumps(args, ensure_ascii=False)}\n[결과]\n{result[:MAX_RESULT_CHARS]}"
        src = [result, json.dumps(args, ensure_ascii=False), subq]
        extra, n, bad = "", None, []
        for _ in range(2):
            n = L.json("note", P.NOTE_SYS, user + extra, NOTE_SCHEMA, v_note, 900)
            if n is None:
                return None, []
            bad = unverified(note_text(n), src)
            if not bad:
                return n, []
            extra = f"\n[고칠 점] 메모의 숫자 {', '.join(bad)}는 결과에 없다. 결과에 적힌 숫자만 그대로 옮겨라."
        return n, bad

    # 3) 점검
    def check(self, question, plan, done, L):
        user = (f"[질문] {question}\n[답에 담을 것]\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(plan["답에 담을 것"], 1))
                + f"\n[단계 메모]\n{notes_text(done)}")
        return L.json("check", self.sys_check, user, check_schema(), v_check, 900)

    # 4) 결합
    def answer(self, question, plan, done, L):
        user = (f"[질문] {question}\n[질문 정리] {plan['질문 정리'] or question}\n"
                f"[가정] {'; '.join(plan['가정']) or '없음'}\n[답에 담을 것]\n"
                + ("\n".join(f"{i}. {x}" for i, x in enumerate(plan["답에 담을 것"], 1)) or "(질문에 맞게)")
                + f"\n[단계 메모]\n{notes_text(done)}")
        text = L.text(self.sys_answer, user)
        if plan["의도"] == ["개념 설명"] and not done:
            return text, []
        src = [question, self.catalog] + [s["결과"] for s in done] + [json.dumps(s["인자"], ensure_ascii=False) for s in done]
        bad = unverified(text, src)
        if bad:
            self.say(f"  [확인] 답에 자료에 없는 숫자 {bad} → 다시 쓰게 함")
            again = L.text(self.sys_answer, user + f"\n\n[고칠 점] 다음 숫자는 메모·결과에 없다: {', '.join(bad)}. "
                                                   f"메모에 있는 숫자만 써서 답을 다시 써라.\n[앞 답]\n{text}")
            bad2 = unverified(again, src)
            if len(bad2) <= len(bad):
                text, bad = again, bad2
        if bad:
            text += "\n\n※ 자료로 확인되지 않은 숫자: " + ", ".join(bad)
        return text, bad

    def run(self, question, on_event=None):
        """질문 하나를 처음부터 처리 (앞 질문과 무관). on_event(종류, 내용)으로 진행을 알림. 결과 전체를 돌려줌"""
        self.emit = on_event or (lambda kind, data: None)
        self.cache = {}
        t0, L = time.time(), LLM()
        run = self.store.new(question)
        self.emit("plan_start", None)
        plan = self.plan(question, L)
        run.save("plan.json", plan)
        self.emit("plan", plan)
        self.say(f"[계획] 의도: {', '.join(plan['의도'])} | {plan['질문 정리']}"
                 + (f" | 가정: {'; '.join(plan['가정'])}" if plan["가정"] else "")
                 + "".join(f"\n  답에 담을 것 {i}. {x}" for i, x in enumerate(plan["답에 담을 것"], 1))
                 + "".join(f"\n  단계 {i}. {s['도구']} {json.dumps(s['인자'], ensure_ascii=False)}"
                           + (" (앞 결과 보고 인자 정함)" if s["앞결과필요"] else "") for i, s in enumerate(plan["단계"], 1)))
        done, bad, chk = [], [], None
        if plan["되물을 것"] and not plan["단계"]:
            answer = plan["되물을 것"]
        else:
            for step in plan["단계"]:
                done.append(self.execute(len(done) + 1, step, done, run, L))
            if done:
                self.emit("check_start", None)
                chk = self.check(question, plan, done, L)
                if chk is not None:
                    have = {step_key(s["도구"], s["인자"]) for s in done}
                    extra = [s for s in chk["추가 단계"] if step_key(s["도구"], s["인자"]) not in have]
                    chk["추가 단계"] = extra            # 이미 한 단계와 같은 것은 뺌
                    run.save("check.json", chk)
                    self.emit("check", chk)
                    self.say(f"  [점검] 빠진 것: {', '.join(chk['빠진 것']) or '없음'} | 추가 단계 {len(extra)}개")
                    for step in extra:
                        done.append(self.execute(len(done) + 1, step, done, run, L))
            self.emit("answer_start", None)
            answer, bad = self.answer(question, plan, done, L)
        meta = {"질문 정리": plan["질문 정리"], "의도": plan["의도"], "단계 수": len(done),
                "오류 단계": sum(s["오류"] for s in done), "확인 안 된 숫자": bad, "LLM 호출": L.calls,
                "LLM 시간(초)": round(L.seconds, 1), "걸린 시간(초)": round(time.time() - t0, 1),
                "모델": config.WRITER_MODEL, "프롬프트 버전": P.PROMPT_VER, "형식 강제": LLM.schema_on}
        run.save_answer(answer, meta)
        self.cache = {}                         # 다음 질문에는 아무것도 넘기지 않음
        self.say(f"  [완료] LLM {L.calls}회, {meta['걸린 시간(초)']}초 → {run.path}")
        return {"질문": question, "답": answer, "계획": plan, "단계": done, "점검": chk, "요약": meta,
                "폴더": str(run.path)}

    def ask(self, question):
        return self.run(question)["답"]


def main():
    ap = argparse.ArgumentParser(description="주식 분석 챗봇")
    ap.add_argument("question", nargs="*", help="질문 (없으면 대화형)")
    ap.add_argument("-v", "--verbose", action="store_true", help="계획과 단계 진행 출력")
    a = ap.parse_args()
    try:
        agent = Agent(verbose=a.verbose)
    except ToolError as e:
        sys.exit(f"준비 실패: {e}")
    try:
        if a.question:
            print(agent.ask(" ".join(a.question)))
            return
        print(f"주식 분석 챗봇 (데이터 마지막 거래일 {tools.last_day()}). 끝내려면 빈 줄에서 Enter")
        while True:
            try:
                q = input("\n질문> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit") or q == "끝":
                break
            print("\n" + agent.ask(q))
    except RuntimeError as e:
        sys.exit(f"\n중단: {e}")


if __name__ == "__main__":
    main()
