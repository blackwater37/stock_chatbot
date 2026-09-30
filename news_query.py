# -*- coding: utf-8 -*-

import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "market.db"
ALIAS_CSV = BASE / "data" / "news_alias.csv"
RECENT = 3      # '최근 사건' = 그날 전 몇 거래일 안에 처음 나온 사건

REL_RANK = {"주인공": 0, "그룹": 1, "업종": 2, "언급": 3, "무관": 4}
REL_W = {"주인공": 1.0, "그룹": 0.8, "업종": 0.6}
TYPE_W = {"사건": 1.0, "시황": 0.9, "리포트": 0.9, "해설": 0.7, "인터뷰": 0.6, "기타": 0.5, "홍보": 0.3, "인사": 0.3}


# ───────────────────────── 종목 별칭 ─────────────────────────
ALIAS_COLS = ["종목코드", "종목명", "별칭", "그룹", "인물", "주요사업"]


def _split(s):
    return [w.strip() for w in re.split(r"[|,/;]", s or "") if w.strip()]


def read_alias():
    """data/news_alias.csv → {종목코드: {name, aliases, group, people, business}}"""
    if not ALIAS_CSV.exists():
        return {}
    out = {}
    with open(ALIAS_CSV, encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            code = (r.get("종목코드") or "").strip().zfill(6)
            out[code] = {"name": (r.get("종목명") or "").strip(), "aliases": _split(r.get("별칭")),
                         "group": _split(r.get("그룹")), "people": _split(r.get("인물")),
                         "business": (r.get("주요사업") or "").strip()}
    return out


def name_words(conn):
    """형태소 분석기에 한 단어로 등록할 이름들 (종목명·별칭·그룹·인물)"""
    words = set()
    try:
        words |= {n for (n,) in conn.execute("SELECT name FROM symbols WHERE kind = 'stock' AND name IS NOT NULL")}
    except sqlite3.OperationalError:
        pass
    for a in read_alias().values():
        words |= {a["name"], *a["aliases"], *a["group"], *a["people"]}
    return {w for w in words if w}


# ───────────────────────── 형태소 토큰 ─────────────────────────
class Tokenizer:
    """Kiwi 형태소 분석으로 뜻 있는 단어(명사·외국어·숫자·동사/형용사 어간)만 남김.
    kiwipiepy가 없으면 한글을 두 글자씩 잘라 대신 씀 (정확도는 조금 낮음)."""
    KEEP = {"NNG", "NNP", "NR", "SL", "SH", "SN", "XR", "VV", "VA"}

    def __init__(self, words=()):
        try:
            from kiwipiepy import Kiwi
        except ImportError:
            self.kiwi, self.name = None, "bigram"
            return
        self.kiwi, self.name = Kiwi(), "kiwi"
        for w in sorted(words):
            if len(w) >= 2 and not re.search(r"\s", w):
                try:
                    self.kiwi.add_user_word(w, "NNP")
                except Exception:
                    pass

    def __call__(self, text):
        text = text or ""
        if self.kiwi:
            return [t.form.lower() for t in self.kiwi.tokenize(text) if t.tag.split("-")[0] in self.KEEP]
        out = []
        for w in re.findall(r"[A-Za-z0-9]+|[가-힣]+", text):
            if re.match(r"[가-힣]", w) and len(w) > 1:
                out += [w[i:i + 2] for i in range(len(w) - 1)]
            else:
                out.append(w.lower())
        return out


def bm25_scores(query_tokens, docs, k1=1.2, b=0.75):
    n = len(docs)
    q = set(query_tokens)
    if not n or not q:
        return [0.0] * n
    lens = [len(d) for d in docs]
    avg = (sum(lens) / n) or 1.0
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d, dl in zip(docs, lens):
        tf = Counter(d)
        s = 0.0
        for t in q:
            f = tf.get(t)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avg))
        out.append(s)
    return out


# ───────────────────────── 조회 함수 ─────────────────────────
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def trading_day(conn, date, after=False):
    """date 이후(after=True면 date 다음) 첫 거래일. 가격 데이터 밖이면 date 그대로"""
    op = ">" if after else ">="
    r = conn.execute(f"SELECT MIN(date) FROM daily_price WHERE code = 'KOSPI' AND date {op} ?", (date,)).fetchone()[0]
    return r or date


def clusters_by_id(conn, ids):
    if not ids:
        return []
    q = f"SELECT * FROM news_cluster WHERE cluster_id IN ({','.join('?' * len(ids))}) ORDER BY first_date, cluster_id"
    return [dict(r) for r in conn.execute(q, list(ids))]


def day_news(conn, code, date):
    """그날(반영 거래일 기준) 뉴스 상태
    new: 그날 처음 반영된 사건 / recent: 직전 RECENT 거래일 안에 나온 사건 / cont: 더 전에 시작해 그날도 보도된 사건
    threads: 위 사건들이 이어받은 예전 사건 (몇 주 전 발표의 후속 등)
    robot: 그날 로봇·시세 기사 제목 / price_notes: 당일·다음 거래일 기사가 말한 주가 움직임 이유"""
    d = trading_day(conn, date)
    out = {"stock_code": code, "date": d, "n_all": 0, "n_event": 0, "base60": 0.0,
           "new": [], "recent": [], "cont": [], "threads": [], "robot": [], "price_notes": []}
    row = conn.execute("SELECT * FROM news_day WHERE stock_code = ? AND date = ?", (code, d)).fetchone()
    if row:
        out.update(n_all=row["n_all"], n_event=row["n_event"], base60=row["base60"])
        for k, col in (("new", "new_clusters"), ("recent", "recent_clusters"), ("cont", "cont_clusters")):
            out[k] = clusters_by_id(conn, json.loads(row[col]))
        for tid in json.loads(row["threads"]):
            th = conn.execute("SELECT * FROM news_thread WHERE thread_id = ?", (tid,)).fetchone()
            if th and th["n_clusters"] > 1:
                shown = {c["cluster_id"] for k in ("new", "recent", "cont") for c in out[k]}
                earlier = [c for c in clusters_by_id(conn, json.loads(th["clusters"]))
                           if c["first_date"] < d and c["cluster_id"] not in shown]
                if earlier:
                    out["threads"].append({"thread_id": tid, "label": th["label"], "earlier": earlier})
    nxt = trading_day(conn, d, after=True)
    out["robot"] = [r[0] for r in conn.execute(
        "SELECT n.title FROM news_article a JOIN news n USING (news_id, stock_code) "
        "WHERE a.stock_code = ? AND a.trade_date = ? AND a.rule_type = 'robot' AND a.dup_of IS NULL", (code, d))]
    out["price_notes"] = [dict(r) for r in conn.execute(
        "SELECT a.trade_date, n.outlet, n.title, a.price_note FROM news_article a JOIN news n USING (news_id, stock_code) "
        "WHERE a.stock_code = ? AND a.trade_date IN (?, ?) AND a.price_note <> '' AND a.dup_of IS NULL "
        "AND a.relevance IN ('주인공', '그룹', '업종', '언급') ORDER BY a.trade_date", (code, d, nxt))]
    return out


def importance(c):
    return (c["n_articles"] * REL_W.get(c["relevance"], 0.3) * TYPE_W.get(c["art_type"], 0.5)
            * (1.3 if c.get("price_note") else 1.0))


def period_news(conn, code, start, end, top=10):
    """기간 안에 처음 나온 사건 중 중요한 것 top개 (날짜순) + 월별 요약"""
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM news_cluster WHERE stock_code = ? AND first_date BETWEEN ? AND ?", (code, start, end))]
    picked = sorted(sorted(rows, key=importance, reverse=True)[:top], key=lambda c: c["first_date"])
    months = [dict(r) for r in conn.execute(
        "SELECT * FROM news_month WHERE stock_code = ? AND month BETWEEN ? AND ? ORDER BY month",
        (code, start[:7], end[:7]))]
    return {"clusters": picked, "n_clusters": len(rows), "months": months}


_embed_warned = False


def search(conn, code, query, start=None, end=None, k=10, tok=None):
    """주제 검색: 종목·기간으로 거른 문서(사건 요약, 월 요약, 기사)에서 키워드(BM25) + 임베딩 순위를 합침"""
    global _embed_warned
    tok = tok or Tokenizer(name_words(conn))
    cond, params = ["stock_code = ?"], [code]
    if start:
        cond.append("date >= ?"); params.append(start)
    if end:
        cond.append("date <= ?"); params.append(end)
    rows = [dict(r) for r in conn.execute(
        f"SELECT doc_type, doc_id, date, text, tokens FROM news_doc WHERE {' AND '.join(cond)}", params)]
    if not rows:
        return []
    bm = bm25_scores(tok(query), [r["tokens"].split() for r in rows])
    rank = Counter()
    for i, idx in enumerate(sorted((j for j in range(len(rows)) if bm[j] > 0), key=lambda j: -bm[j])):
        rank[idx] += 1 / (60 + i)
    sims = {}
    try:
        import llm
        qv = np.asarray(llm.embed([query])[0], dtype=np.float32)
        qv /= np.linalg.norm(qv) or 1
        pos = {(r["doc_type"], r["doc_id"]): j for j, r in enumerate(rows)}
        for t, i, v in conn.execute("SELECT doc_type, doc_id, vec FROM news_vec WHERE stock_code = ?", (code,)):
            if (t, i) in pos:
                sims[pos[(t, i)]] = float(np.frombuffer(v, dtype=np.float32) @ qv)
        for i, idx in enumerate(sorted(sims, key=lambda j: -sims[j])[:50]):
            rank[idx] += 1 / (60 + i)
    except Exception as e:
        if not _embed_warned:
            print(f"  (임베딩 검색 생략, 키워드 검색만: {str(e)[:80]})")
            _embed_warned = True
    out = []
    for idx, score in rank.most_common(k):
        r = rows[idx]
        out.append({"doc_type": r["doc_type"], "doc_id": r["doc_id"], "date": r["date"], "text": r["text"],
                    "score": round(score, 4), "bm25": round(bm[idx], 2), "sim": round(sims.get(idx, 0.0), 3)})
    return out


# ───────────────────────── 화면 출력 ─────────────────────────
def _c(c):
    note = f" / 주가설명: {c['price_note']}" if c.get("price_note") else ""
    return (f"  {c['first_date']} {c['label']} (기사 {c['n_articles']}건, {c['relevance']}, {c['art_type']}, "
            f"{c['direction']}, {c['certainty']})\n      {c['summary']}{note}")


def print_day(conn, code, date):
    r = day_news(conn, code, date)
    base = f", 최근 60거래일 평균 {r['base60']:.1f}건" if r["base60"] else ""
    print(f"{code} {r['date']} 반영 뉴스: 전체 {r['n_all']}건, 사건 기사 {r['n_event']}건{base}")
    for key, title in (("new", "그날 처음 나온 사건"), ("recent", f"직전 {RECENT}거래일 사건"),
                       ("cont", "더 전에 시작해 그날도 보도된 사건")):
        print(f"\n[{title}]")
        print("\n".join(_c(c) for c in r[key]) if r[key] else "  없음")
    if r["threads"]:
        print("\n[이어지는 예전 사건 (사건 줄기)]")
        for t in r["threads"]:
            print(f"  줄기 '{t['label']}': " + " → ".join(f"{c['first_date']} {c['label']}" for c in t["earlier"]))
    if r["robot"]:
        print("\n[그날 로봇·시세 기사]")
        print("\n".join(f"  {t}" for t in r["robot"]))
    if r["price_notes"]:
        print("\n[주가 움직임 이유를 말한 기사 (당일·다음 거래일)]")
        print("\n".join(f"  {p['trade_date']} {p['outlet']} | {p['title']} → {p['price_note']}" for p in r["price_notes"]))


def print_period(conn, code, start, end):
    r = period_news(conn, code, start, end)
    print(f"{code} {start} ~ {end}: 사건 {r['n_clusters']}개 중 중요한 {len(r['clusters'])}개")
    print("\n".join(_c(c) for c in r["clusters"]) or "  없음")
    if r["months"]:
        print("\n[월별 요약]")
        for m in r["months"]:
            print(f"  {m['month']} ({m['tone']}, 사건 {m['n_clusters']}개): {m['summary']}")


def print_search(conn, code, query, start=None, end=None):
    res = search(conn, code, query, start, end)
    print(f"{code} '{query}' 검색 {len(res)}건")
    for r in res:
        print(f"  [{r['doc_type']}] {r['date']} (키워드 {r['bm25']}, 의미 {r['sim']}) {r['text'][:120]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3 or a[0] not in ("day", "period", "search"):
        sys.exit(__doc__)
    con = connect()
    if a[0] == "day":
        print_day(con, a[1], a[2])
    elif a[0] == "period":
        print_period(con, a[1], a[2], a[3] if len(a) > 3 else a[2])
    else:
        print_search(con, a[1], a[2], *(a[3:5]))
