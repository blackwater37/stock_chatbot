# -*- coding: utf-8 -*-
import json
import re
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUN_DIR = BASE / "data" / "agent_runs"


class Run:
    """실행 하나의 저장소 (폴더)"""
    def __init__(self, path, question):
        self.path, self.question = path, question
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, name, obj):
        with open(self.path / name, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False, indent=2)

    def load(self, name, default=None):
        p = self.path / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def save_step(self, no, rec):
        self.save(f"step_{no:02d}.json", rec)

    def save_answer(self, text, meta):
        (self.path / "answer.md").write_text(f"# 질문\n{self.question}\n\n# 답\n{text}\n", encoding="utf-8")
        self.save("run.json", {"질문": self.question, "시각": self.path.name[:15], **meta})


class RunStore:
    def __init__(self, root=RUN_DIR):
        self.root = Path(root)

    def new(self, question):
        slug = re.sub(r'[\\/:*?"<>|\s]+', " ", question).strip()[:30] or "질문"
        base = f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
        path, n = self.root / base, 2
        while path.exists():
            path, n = self.root / f"{base}_{n}", n + 1
        return Run(path, question)

    def list(self, n=20):
        if not self.root.exists():
            return []
        dirs = sorted((p for p in self.root.iterdir() if p.is_dir()), reverse=True)[:n]
        return [(p, Run(p, "").load("run.json", {})) for p in dirs]


def show(path):
    """실행 하나를 사람이 읽기 좋게 출력"""
    run = Run(Path(path), "")
    plan = run.load("plan.json", {})
    meta = run.load("run.json", {})
    print(f"■ {meta.get('질문', '')}")
    print(f"  의도: {', '.join(plan.get('의도', []))} | 질문 정리: {plan.get('질문 정리', '')}")
    if plan.get("가정"):
        print(f"  가정: {', '.join(plan['가정'])}")
    if plan.get("되물을 것"):
        print(f"  되물음: {plan['되물을 것']}")
    for i, x in enumerate(plan.get("답에 담을 것", []), 1):
        print(f"  답에 담을 것 {i}. {x}")
    for p in sorted(run.path.glob("step_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        print(f"\n[단계 {s['번호']}] {s['소질문']}\n  {s['도구']} {json.dumps(s['인자'], ensure_ascii=False)}"
              + (" (앞 질문 결과 재사용)" if s.get("재사용") else ""))
        note = s.get("메모")
        if s.get("오류"):
            print(f"  오류: {s['결과'][:200]}")
        elif note:
            print(f"  결론: {note.get('결론', '')}")
            if note.get("핵심 숫자"):
                print(f"  핵심 숫자: {'; '.join(note['핵심 숫자'])}")
            if note.get("근거"):
                print(f"  근거: {'; '.join(note['근거'])}")
            if note.get("빈 곳"):
                print(f"  빈 곳: {note['빈 곳']}")
        else:
            print(f"  결과(원본): {s['결과'][:300]}")
        if s.get("확인 안 된 숫자"):
            print(f"  ※ 메모에서 확인 안 된 숫자: {s['확인 안 된 숫자']}")
    chk = run.load("check.json")
    if chk:
        print(f"\n[점검] 빠진 것: {', '.join(chk.get('빠진 것', [])) or '없음'}"
              f" | 추가 단계: {', '.join(s['도구'] for s in chk.get('추가 단계', [])) or '없음'}")
    ans = run.path / "answer.md"
    if ans.exists():
        print("\n" + ans.read_text(encoding="utf-8").split("# 답\n", 1)[-1].strip())
    if meta:
        print(f"\n(LLM {meta.get('LLM 호출', '?')}회, {meta.get('걸린 시간(초)', '?')}초, {meta.get('모델', '')}, "
              f"{meta.get('프롬프트 버전', '')})")


if __name__ == "__main__":
    store = RunStore()
    runs = store.list(50)
    if len(sys.argv) < 2:
        if not runs:
            sys.exit("저장된 실행이 없습니다.")
        for i, (p, m) in enumerate(runs[:20], 1):
            print(f"{i:>2}. {p.name[:15]}  단계 {m.get('단계 수', '?')}  LLM {m.get('LLM 호출', '?')}회  "
                  f"{m.get('걸린 시간(초)', '?')}초  {m.get('질문', p.name[16:])[:50]}")
    else:
        a = sys.argv[1]
        target = runs[int(a) - 1][0] if a.isdigit() and 0 < int(a) <= len(runs) else RUN_DIR / a
        if not target.exists():
            sys.exit(f"없는 실행: {a}")
        show(target)
