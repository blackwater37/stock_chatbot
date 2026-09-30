# -*- coding: utf-8 -*-

import json
import math
import re
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import news_query

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "data" / "market.db"

TOOLS = {}          # 도구 이름 → {"fn": 함수, "desc": 설명, "args": 인자 설명, "example": 예시 인자}


class ToolError(Exception):
    """LLM에게 그대로 돌려줄 오류 (인자를 고쳐 다시 요청하게)"""


def arg(desc, required=False, kind="str", choices=None):
    return {"desc": desc, "required": required, "kind": kind, "choices": choices}


def tool(name, desc, args, example):
    """함수를 도구로 등록 (agent.py 가 이 정보로 프롬프트와 JSON 형식을 만든다)"""
    def deco(fn):
        TOOLS[name] = {"fn": fn, "desc": desc, "args": args, "example": example}
        return fn
    return deco


# ───────────────────────── DB·달력 ─────────────────────────
_conn = _cal = _pos = _names = _alias = _tok = None


def db():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA query_only = ON")       # 도구는 읽기만
    return _conn


def calendar():
    """거래일 목록 (코스피 지수 날짜)"""
    global _cal, _pos
    if _cal is None:
        _cal = [d for (d,) in db().execute("SELECT date FROM daily_price WHERE code = 'KOSPI' ORDER BY date")]
        if not _cal:
            raise ToolError("코스피 지수 가격이 없습니다 (collect_prices.py 필요)")
        _pos = {d: i for i, d in enumerate(_cal)}
    return _cal


def last_day():
    return calendar()[-1]


def td_after(d):
    """d 이후(포함) 첫 거래일"""
    cal = calendar()
    i = bisect_left(cal, d)
    return cal[i] if i < len(cal) else None


def td_before(d):
    """d 이전(포함) 마지막 거래일"""
    cal = calendar()
    i = bisect_right(cal, d) - 1
    return cal[i] if i >= 0 else None


def td_shift(d, n):
    """거래일 d에서 n거래일 이동 (데이터 끝을 넘으면 끝)"""
    cal = calendar()
    return cal[min(max(_pos[d] + n, 0), len(cal) - 1)]


def trading_range(s, e):
    s0, e0 = td_after(s), td_before(e)
    if not s0 or not e0 or s0 > e0:
        raise ToolError(f"{s}~{e} 사이에 거래일이 없습니다 (가격 데이터 {calendar()[0]}~{last_day()})")
    return s0, e0


def data_catalog():
    """LLM에게 알려 줄 데이터 목록 (있는 것·없는 것)"""
    c = db()
    lines = []
    n, d0, d1 = c.execute("SELECT COUNT(*), MIN(first_date), MAX(last_date) FROM symbols WHERE kind = 'stock'").fetchone()
    lines.append(f"- 가격: 코스피 보통주 {n}종목 일별 시가·고가·저가·종가·거래량 ({d0}~{d1}), 코스피·코스닥 지수")
    try:
        nf = c.execute("SELECT COUNT(DISTINCT stock_code) FROM fs_item").fetchone()[0]
        lines.append(f"- 재무: {nf}종목 분기 재무제표 (연결 우선, 없으면 별도). 손익·현금흐름은 분기(3개월)·누적·연간")
    except sqlite3.OperationalError:
        lines.append("- 재무: 없음")
    try:
        rows = c.execute("SELECT stock_code, MIN(pub_date), MAX(pub_date) FROM news WHERE stock_code IN "
                         "(SELECT DISTINCT stock_code FROM news_cluster) GROUP BY stock_code").fetchall()
        lines.append("- 뉴스 색인: " + (", ".join(f"{names().get(k, k)}({k}) {a}~{b}" for k, a, b in rows) or "없음")
                     + " (다른 종목·기간은 뉴스 없음 → 기술적 분석만)")
    except sqlite3.OperationalError:
        lines.append("- 뉴스 색인: 없음 (기술적 분석만)")
    lines.append("- 없음: 외국인·기관 수급, 시가총액·PER·PBR, 거래대금, 공시 원문, 업종 분류, 증권사 컨센서스")
    return "\n".join(lines)


# ───────────────────────── 기간 해석 ─────────────────────────
PERIOD_HELP = ("기간 형식: 2026-04-30 / 2026-04-01~2026-06-30 / 2026-04 / 2025Q3 / 2025년 / "
               "최근 3개월·최근 20거래일·최근 1년 / 2026-06-09부터 1개월 / 2026-06-09 이전 1개월 / "
               "올해 / 작년 / 이번달 / 지난달")
UNITS = r"(거래일|일|주|개월|달|년)"


def _span(a, n, u, forward):
    """날짜 a에서 n 단위만큼 앞(forward) 또는 뒤로 간 날짜"""
    if u == "거래일":
        d = td_after(a.isoformat()) if forward else td_before(a.isoformat())
        if d is None:
            raise ToolError(f"{a}는 가격 데이터 범위 밖입니다")
        return date.fromisoformat(td_shift(d, n if forward else -(n - 1)))
    k = n if forward else -n
    return {"일": a + timedelta(days=k), "주": a + timedelta(weeks=k), "개월": _add_months(a, k),
            "달": _add_months(a, k), "년": _add_months(a, 12 * k)}[u]


def _d(y, m, dd):
    try:
        return date(int(y), int(m), int(dd))
    except ValueError:
        raise ToolError(f"없는 날짜: {y}-{m}-{dd}")


def _add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    return date(y, m + 1, min(d.day, monthrange(y, m + 1)[1]))


def parse_period(expr):
    """기간 표현 → (시작, 끝, 하루인지). 날짜 계산은 여기서만 한다 (기준일 = 데이터 마지막 거래일)"""
    s = re.sub(r"\s+", "", str(expr or ""))
    if not s:
        raise ToolError("기간이 없습니다. " + PERIOD_HELP)
    s = s.replace("부터", "~").replace("까지", "").replace("/", "-").replace(".", "-").strip("-")
    s = re.sub(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", r"\1-\2-\3", s)
    ref = date.fromisoformat(last_day())
    D = r"(\d{4})-(\d{1,2})-(\d{1,2})"
    if m := re.fullmatch(D + r"[~+](\d+)" + UNITS + r"(?:간|동안)?", s):          # 2026-06-09부터 1개월
        a = _d(*m.groups()[:3])
        return a.isoformat(), min(_span(a, int(m[4]), m[5], True), ref).isoformat(), False
    if m := re.fullmatch(D + r"(?:이전|전)(\d+)" + UNITS + r"(?:간|동안)?", s):    # 2026-06-09 이전 1개월
        end = _d(*m.groups()[:3]) - timedelta(days=1)
        return _span(end, int(m[4]), m[5], False).isoformat(), end.isoformat(), False
    if m := re.fullmatch(D + "~" + D, s):
        a, b = _d(*m.groups()[:3]), _d(*m.groups()[3:])
        return min(a, b).isoformat(), max(a, b).isoformat(), False
    if m := re.fullmatch(D + "~", s):
        return _d(*m.groups()).isoformat(), ref.isoformat(), False
    if m := (re.fullmatch(D, s) or re.fullmatch(r"(\d{4})년(\d{1,2})월(\d{1,2})일", s)):
        x = _d(*m.groups()).isoformat()
        return x, x, True
    if m := (re.fullmatch(r"(\d{4})-(\d{1,2})", s) or re.fullmatch(r"(\d{4})년(\d{1,2})월", s)):
        y, mo = int(m[1]), int(m[2])
        _d(y, mo, 1)
        return date(y, mo, 1).isoformat(), date(y, mo, monthrange(y, mo)[1]).isoformat(), False
    if m := (re.fullmatch(r"(\d{4})[Qq]([1-4])", s) or re.fullmatch(r"(\d{4})년([1-4])분기", s)):
        y, q = int(m[1]), int(m[2])
        return date(y, 3 * q - 2, 1).isoformat(), date(y, 3 * q, monthrange(y, 3 * q)[1]).isoformat(), False
    if m := re.fullmatch(r"(\d{4})년?", s):
        return f"{m[1]}-01-01", f"{m[1]}-12-31", False
    if m := re.fullmatch(r"(?:최근|지난)(\d+)(거래일|일|주|개월|달|년)", s):
        n, u = int(m[1]), m[2]
        if u == "거래일":
            cal = calendar()
            return cal[max(0, len(cal) - n)], ref.isoformat(), False
        start = {"일": ref - timedelta(days=n - 1), "주": ref - timedelta(days=7 * n - 1),
                 "개월": _add_months(ref, -n) + timedelta(days=1), "달": _add_months(ref, -n) + timedelta(days=1),
                 "년": _add_months(ref, -12 * n) + timedelta(days=1)}[u]
        return start.isoformat(), ref.isoformat(), False
    first = ref.replace(day=1)
    prev_last = first - timedelta(days=1)
    words = {"오늘": (ref, ref), "올해": (date(ref.year, 1, 1), ref), "금년": (date(ref.year, 1, 1), ref),
             "작년": (date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)),
             "지난해": (date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31)),
             "이번달": (first, ref), "지난달": (prev_last.replace(day=1), prev_last)}
    if s in words:
        a, b = words[s]
        return a.isoformat(), b.isoformat(), a == b
    raise ToolError(f"기간을 알 수 없음: {expr}. " + PERIOD_HELP)


# ───────────────────────── 종목 찾기 ─────────────────────────
INDEX = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}


def _norm(s):
    return re.sub(r"\s|㈜|\(주\)|주식회사", "", str(s or "")).lower()


def names():
    global _names
    if _names is None:
        _names = {c: n for c, n in db().execute("SELECT code, name FROM symbols WHERE kind = 'stock'")}
    return _names


def aliases():
    global _alias
    if _alias is None:
        _alias = {c: {_norm(w) for w in [a["name"], *a["aliases"]] if w} for c, a in news_query.read_alias().items()}
    return _alias


def resolve(x):
    """이름·약칭·코드 → (코드, 이름). 여러 개가 걸리면 후보를 ToolError로"""
    s = str(x or "").strip()
    if not s:
        raise ToolError("종목이 없습니다")
    for code, kor in INDEX.items():
        if s.upper() == code or s in (kor, kor + "지수"):
            return code, kor
    nm = names()
    if s.upper() in nm:
        return s.upper(), nm[s.upper()]
    if re.fullmatch(r"\d{6}", s):
        raise ToolError(f"코드 {s} 종목의 데이터가 없습니다 (코스피 보통주만 있음)")
    key = _norm(s)
    exact = [c for c, n in nm.items() if _norm(n) == key] + [c for c, al in aliases().items() if key in al and c in nm]
    if exact:
        return exact[0], nm[exact[0]]
    part = [c for c, n in nm.items() if key in _norm(n)]
    if len(part) == 1:
        return part[0], nm[part[0]]
    if part:
        raise ToolError("여러 종목이 해당됩니다: " + ", ".join(f"{nm[c]}({c})" for c in part[:10])
                        + " → 정확한 이름이나 코드로 다시 요청")
    raise ToolError(f"종목을 찾지 못함: {s} (코스피 보통주만 있음)")


def resolve_many(x, limit=5):
    items = [resolve(p) for p in re.split(r"[,/]", str(x or "")) if p.strip()]
    if not items:
        raise ToolError("종목이 없습니다")
    if len(items) > limit:
        raise ToolError(f"종목은 한 번에 {limit}개까지")
    return list(dict.fromkeys(items))


# ───────────────────────── 가격 계산 도우미 ─────────────────────────
def prices(codes, start, end, lookback=0, lookahead=0):
    """{코드: 일별 가격 표}. 코스피(KOSPI)는 항상 포함. start 앞 lookback 거래일, end 뒤 lookahead 거래일까지"""
    cal = calendar()
    i0 = max(0, bisect_left(cal, start) - lookback)
    i1 = min(len(cal) - 1, bisect_right(cal, end) - 1 + lookahead)
    codes = list(dict.fromkeys([*codes, "KOSPI"]))
    q = (f"SELECT code, date, open, high, low, close, volume FROM daily_price "
         f"WHERE code IN ({','.join('?' * len(codes))}) AND date BETWEEN ? AND ? ORDER BY date")
    rows = [tuple(r) for r in db().execute(q, [*codes, cal[i0], cal[max(i0, i1)]])]
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close", "volume"])
    out = {}
    for c, g in df.groupby("code"):
        t = g.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
        out[c] = t[t["close"] > 0]
    return out


def series(px, code, name):
    if code not in px or px[code].empty:
        raise ToolError(f"{name}({code}) 가격 데이터가 없습니다")
    return px[code]


def beta(r, m, before, n=250):
    """before 전 n거래일 일별 수익률로 구한 지수 민감도 (자료가 적으면 1)"""
    x = pd.concat([r, m], axis=1, keys=["r", "m"]).dropna()
    x = x[x.index < before].tail(n)
    if len(x) < 60 or x["m"].var() == 0:
        return 1.0
    return float(np.clip(x["r"].cov(x["m"]) / x["m"].var(), -1, 3))


def _nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, sign=True):
    return "없음" if _nan(x) else (f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%")


def pp(x):
    return "없음" if _nan(x) else f"{x * 100:+.1f}%p"


def won(v):
    if _nan(v):
        return "없음"
    a, sgn = abs(v), "-" if v < 0 else ""
    if a >= 1e12:
        return f"{sgn}{a / 1e12:,.2f}조원"
    if a >= 1e8:
        return f"{sgn}{a / 1e8:,.0f}억원"
    if a >= 1e4:
        return f"{sgn}{a / 1e4:,.0f}만원"
    return f"{v:,.0f}원"


def _day(code, name, d):
    """하루 움직임 → (보여 줄 표, 계산값)"""
    d0 = td_after(d)
    if d0 is None:
        raise ToolError(f"{d}는 가격 데이터 범위(~{last_day()}) 밖입니다")
    px = prices([code], d0, d0, lookback=260, lookahead=5)
    s, k = series(px, code, name), px["KOSPI"]["close"]
    if d0 not in s.index:
        raise ToolError(f"{name}은(는) {d0}에 거래 기록이 없습니다 (거래정지 등)")
    i = s.index.get_loc(d0)
    if i == 0:
        raise ToolError(f"{name}의 {d0} 이전 가격이 없어 등락을 계산할 수 없습니다")
    c, prev_d = s["close"], s.index[i - 1]
    prev = c.iloc[i - 1]
    ret = c.iloc[i] / prev - 1
    kret = k[d0] / k[prev_d] - 1 if d0 in k.index and prev_d in k.index else float("nan")
    b = beta(c.pct_change(), k.pct_change(), d0)
    mk = b * kret
    vbase = s["volume"].iloc[max(0, i - 20):i].mean()
    vr = s["volume"].iloc[i] / vbase if vbase > 0 else float("nan")
    out = {"종목": f"{name}({code})", "날짜": d0, "종가": f"{c.iloc[i]:,.0f}원", "등락률": pct(ret), "코스피": pct(kret),
           "시장 몫": f"{pp(mk)} (베타 {b:.2f})", "고유 등락": pp(ret - mk),
           "거래량": "없음" if _nan(vr) else f"직전 20일 평균의 {vr:.1f}배",
           "시가 갭": pct(s["open"].iloc[i] / prev - 1),
           "장중 고가·저가": f"{pct(s['high'].iloc[i] / prev - 1)} / {pct(s['low'].iloc[i] / prev - 1)}",
           "직전 5거래일": pct(prev / c.iloc[i - 6] - 1) if i >= 6 else "없음",
           "이후 5거래일": pct(c.iloc[i + 5] / c.iloc[i] - 1) if i + 5 < len(c) else "없음(데이터 끝)"}
    if abs(ret) >= 0.295:
        out["비고"] = "상한가" if ret > 0 else "하한가"
    if d0 != d:
        out["날짜 참고"] = f"{d}는 휴장일이라 다음 거래일 {d0} 기준"
    return out, {"date": d0, "ret": ret, "kret": kret, "beta": b, "own": ret - mk, "vol_ratio": vr}


def _period(code, name, s, e):
    """기간 움직임 → (보여 줄 표, 계산값)"""
    s0, e0 = trading_range(s, e)
    px = prices([code], s0, e0, lookback=260)
    st, k = series(px, code, name), px["KOSPI"]["close"]
    c = st["close"]
    win = c[(c.index >= s0) & (c.index <= e0)]
    if win.empty:
        raise ToolError(f"{name}의 {s0}~{e0} 거래 기록이 없습니다")
    before = c[c.index < win.index[0]]
    if len(before):
        base, base_d = before.iloc[-1], before.index[-1]
        path = pd.concat([before.tail(1), win])
    else:
        base, base_d, path = win.iloc[0], win.index[0], win
    ret = win.iloc[-1] / base - 1
    kret = k[win.index[-1]] / k[base_d] - 1 if win.index[-1] in k.index and base_d in k.index else float("nan")
    b = beta(c.pct_change(), k.pct_change(), win.index[0])
    r = path.pct_change().dropna()
    ab = r - b * k.pct_change().reindex(r.index)
    vol = st["volume"]
    vb = vol[vol.index < win.index[0]].tail(60).mean()
    vw = vol[win.index].mean()
    out = {"종목": f"{name}({code})", "기간": f"{win.index[0]}~{win.index[-1]} ({len(win)}거래일)",
           "기준가": f"{base:,.0f}원 ({base_d} 종가)", "마지막 종가": f"{win.iloc[-1]:,.0f}원",
           "수익률": pct(ret), "코스피": pct(kret), "코스피 대비": pp(ret - kret),
           "베타 반영 고유 수익": f"{pp(ret - b * kret)} (베타 {b:.2f})",
           "최고": f"{win.max():,.0f}원 ({win.idxmax()})", "최저": f"{win.min():,.0f}원 ({win.idxmin()})",
           "최대 낙폭": pct((path / path.cummax() - 1).min()),
           "변동성(연환산)": pct(r.std() * math.sqrt(250), sign=False) if len(r) > 5 else "없음",
           "거래량": f"직전 60일 평균의 {vw / vb:.1f}배" if vb > 0 else "없음",
           "상승일·하락일": f"{int((r > 0).sum())}일 / {int((r < 0).sum())}일"}
    if ab.notna().any():
        hi, lo = ab.idxmax(), ab.idxmin()
        out["가장 큰 상승일"] = f"{hi} ({pct(r[hi])}, 고유 {pp(ab[hi])})"
        out["가장 큰 하락일"] = f"{lo} ({pct(r[lo])}, 고유 {pp(ab[lo])})"
    if not len(before):
        out["참고"] = "기간 시작 전 가격이 없어 첫날 종가 기준"
    return out, {"s0": win.index[0], "e0": win.index[-1], "ret": ret, "kret": kret, "beta": b,
                 "own": ret - b * kret, "vol_ratio": vw / vb if vb > 0 else float("nan")}


def _tech(code, name, d):
    """d(포함) 기준 기술적 상태"""
    d0 = td_before(d)
    if d0 is None:
        raise ToolError(f"{d} 이전 가격이 없습니다")
    px = prices([code], d0, d0, lookback=260)
    st = series(px, code, name)
    st = st[st.index <= d0]
    c, v = st["close"], st["volume"]
    last = c.iloc[-1]
    ma = {n: c.tail(n).mean() for n in (5, 20, 60, 120) if len(c) >= n}
    if len(ma) == 4 and ma[5] > ma[20] > ma[60] > ma[120]:
        arr = "정배열(상승 추세)"
    elif len(ma) == 4 and ma[5] < ma[20] < ma[60] < ma[120]:
        arr = "역배열(하락 추세)"
    else:
        arr = "혼조"
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    rsi = 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)
    w = c.tail(250)
    hi, lo = w.max(), w.min()
    v60 = v.tail(60).mean()
    return {"종목": f"{name}({code})", "기준일": c.index[-1], "종가": f"{last:,.0f}원",
            "이동평균": ", ".join(f"{n}일선 {'위' if last >= m else '아래'}({pct(last / m - 1)})" for n, m in ma.items()),
            "배열": arr, "20일 이격도": f"{last / ma[20] * 100:.1f}" if 20 in ma else "없음",
            "RSI(14)": f"{rsi:.0f}" + (" (과열권)" if rsi >= 70 else " (침체권)" if rsi <= 30 else ""),
            "52주 고가": f"{hi:,.0f}원 ({w.idxmax()}), 현재가는 {pct(last / hi - 1)}",
            "52주 저가": f"{lo:,.0f}원 ({w.idxmin()}), 현재가는 {pct(last / lo - 1)}",
            "52주 범위 안 위치": f"{(last - lo) / (hi - lo) * 100:.0f}%" if hi > lo else "없음",
            "신고가": "52주 신고가" if last >= hi else ("최근 5거래일 안에 52주 신고가" if c.tail(5).max() >= hi else "아님"),
            "거래량 추세": f"최근 5일 평균이 60일 평균의 {v.tail(5).mean() / v60:.1f}배" if v60 > 0 else "없음",
            "직전 20거래일 수익률": pct(last / c.iloc[-21] - 1) if len(c) > 20 else "없음",
            "직전 60거래일 수익률": pct(last / c.iloc[-61] - 1) if len(c) > 60 else "없음"}


def _rsi(c, n=14):
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100.0)


def _day_signals(code, name, d0):
    """그날 가격·거래량에서 보이는 기술적 신호 (뉴스 없이도 설명에 쓸 수 있는 것)"""
    px = prices([code], d0, d0, lookback=260)
    st = series(px, code, name)
    st = st[st.index <= d0]
    c, o, h, lo, v = (st[k] for k in ("close", "open", "high", "low", "volume"))
    if len(c) < 3 or c.index[-1] != d0:
        return ["신호를 계산할 가격이 부족함"]
    last, prev = c.iloc[-1], c.iloc[-2]
    out = []
    for n in (5, 20, 60, 120):                              # 이동평균선 돌파·이탈
        if len(c) > n:
            now, before = c.iloc[-n:].mean(), c.iloc[-n - 1:-1].mean()
            if prev < before and last >= now:
                out.append(f"{n}일 이동평균선 상향 돌파")
            elif prev >= before and last < now:
                out.append(f"{n}일 이동평균선 하향 이탈")
    past = c.iloc[-251:-1]                                   # 52주 고가·저가 (당일 제외)
    if len(past) >= 60:
        if last > past.max():
            out.append(f"52주 신고가 경신 (이전 최고 {past.max():,.0f}원)")
        elif last < past.min():
            out.append(f"52주 신저가 경신 (이전 최저 {past.min():,.0f}원)")
        elif last >= past.max() * 0.97:
            out.append(f"52주 고가에 근접 ({pct(last / past.max() - 1)})")
    gap = o.iloc[-1] / prev - 1
    if abs(gap) >= 0.02:
        out.append(f"갭 {'상승' if gap > 0 else '하락'} 출발 ({pct(gap)})")
    rng = h.iloc[-1] - lo.iloc[-1]
    if rng > 0:                                              # 봉 모양
        body, pos = (last - o.iloc[-1]) / prev, (last - lo.iloc[-1]) / rng
        if body >= 0.03 and pos >= 0.8:
            out.append("장대 양봉 (고가 부근에서 마감)")
        elif body <= -0.03 and pos <= 0.2:
            out.append("장대 음봉 (저가 부근에서 마감)")
        elif last > prev and pos <= 0.4:
            out.append("올랐지만 고가에서 많이 밀림 (윗꼬리)")
        elif last < prev and pos >= 0.6:
            out.append("내렸지만 저가에서 많이 회복 (아랫꼬리)")
    d = np.sign(c.diff().dropna().to_numpy())               # 연속 상승·하락
    k = 0
    for x in d[::-1]:
        if x == d[-1] and x != 0:
            k += 1
        else:
            break
    if k >= 3:
        out.append(f"{k}거래일 연속 {'상승' if d[-1] > 0 else '하락'}")
    if len(c) > 6:                                           # 급락 뒤 반등, 급등 뒤 조정
        pre5 = prev / c.iloc[-7] - 1
        if last > prev and pre5 <= -0.08:
            out.append(f"직전 5거래일 {pct(pre5)} 급락 뒤 반등")
        elif last < prev and pre5 >= 0.08:
            out.append(f"직전 5거래일 {pct(pre5)} 급등 뒤 조정")
    base = v.iloc[-21:-1].mean()
    if base > 0:
        r = v.iloc[-1] / base
        if r >= 2:
            out.append(f"거래량 급증 (직전 20일 평균의 {r:.1f}배)")
        elif r <= 0.5:
            out.append(f"거래량 적음 (직전 20일 평균의 {r:.1f}배)")
    rs = _rsi(c)
    a, b = rs.iloc[-2], rs.iloc[-1]
    if a < 30 <= b:
        out.append(f"RSI 과매도권 탈출 ({a:.0f} → {b:.0f})")
    elif b >= 70:
        out.append(f"RSI 과열권 ({b:.0f})")
    elif b <= 30:
        out.append(f"RSI 과매도권 ({b:.0f})")
    return out or ["눈에 띄는 기술적 신호 없음"]


def _period_signals(code, name, s0, e0):
    """기간 동안의 기술적 변화: 신고가·신저가, 20일선 돌파·이탈, 시작 전과 끝의 상태 비교, 거래량"""
    px = prices([code], s0, e0, lookback=260)
    st = series(px, code, name)
    c, v = st["close"], st["volume"]
    idx = [i for i, d in enumerate(c.index) if s0 <= d <= e0]
    if not idx:
        return ["신호를 계산할 가격이 부족함"]
    i0, i1 = idx[0], idx[-1]
    out = []
    hi_n = lo_n = 0
    for i in idx:
        past = c.iloc[max(0, i - 250):i]
        if len(past) >= 60:
            hi_n += c.iloc[i] > past.max()
            lo_n += c.iloc[i] < past.min()
    if hi_n:
        out.append(f"기간 중 52주 신고가 경신 {int(hi_n)}번")
    if lo_n:
        out.append(f"기간 중 52주 신저가 경신 {int(lo_n)}번")
    ma = c.rolling(20).mean()
    ups = downs = 0
    for i in idx:
        if i > 0 and pd.notna(ma.iloc[i - 1]) and pd.notna(ma.iloc[i]):
            was, now = c.iloc[i - 1] >= ma.iloc[i - 1], c.iloc[i] >= ma.iloc[i]
            ups += (not was) and now
            downs += was and (not now)
    out.append(f"20일 이동평균선 상향 돌파 {int(ups)}번, 하향 이탈 {int(downs)}번")
    before = _tech(code, name, c.index[i0 - 1] if i0 > 0 else c.index[i0])
    after = _tech(code, name, c.index[i1])
    for k in ("배열", "RSI(14)", "52주 범위 안 위치"):
        out.append(f"{k}: 기간 전 {before[k]} → 기간 끝 {after[k]}")
    vb, vw = v.iloc[max(0, i0 - 60):i0].mean(), v.iloc[i0:i1 + 1].mean()
    if vb > 0:
        out.append(f"기간 평균 거래량은 직전 60일 평균의 {vw / vb:.1f}배")
    return out


# ───────────────────────── 재무 계산 도우미 ─────────────────────────
METRICS = {
    "매출": ("PL", "ifrs-full_Revenue"), "매출원가": ("PL", "ifrs-full_CostOfSales"),
    "매출총이익": ("PL", "ifrs-full_GrossProfit"), "영업이익": ("PL", "dart_OperatingIncomeLoss"),
    "순이익": ("PL", "ifrs-full_ProfitLoss"), "지배순이익": ("PL", "ifrs-full_ProfitLossAttributableToOwnersOfParent"),
    "자산": ("BS", "ifrs-full_Assets"), "부채": ("BS", "ifrs-full_Liabilities"), "자본": ("BS", "ifrs-full_Equity"),
    "유동자산": ("BS", "ifrs-full_CurrentAssets"), "유동부채": ("BS", "ifrs-full_CurrentLiabilities"),
    "현금": ("BS", "ifrs-full_CashAndCashEquivalents"),
    "영업현금흐름": ("CF", "ifrs-full_CashFlowsFromUsedInOperatingActivities"),
    "투자현금흐름": ("CF", "ifrs-full_CashFlowsFromUsedInInvestingActivities"),
    "재무현금흐름": ("CF", "ifrs-full_CashFlowsFromUsedInFinancingActivities"),
}
RATIOS = {"영업이익률": ("영업이익", "매출"), "순이익률": ("순이익", "매출"), "매출총이익률": ("매출총이익", "매출"),
          "부채비율": ("부채", "자본"), "유동비율": ("유동자산", "유동부채")}
METRIC_ALIAS = {"매출액": "매출", "영업수익": "매출", "수익": "매출", "영업익": "영업이익", "영업손익": "영업이익",
                "영업손실": "영업이익", "당기순이익": "순이익", "순손익": "순이익", "당기순손익": "순이익", "순손실": "순이익",
                "지배주주순이익": "지배순이익", "지배순익": "지배순이익", "총자산": "자산", "자산총계": "자산",
                "부채총계": "부채", "자본총계": "자본", "현금성자산": "현금", "현금및현금성자산": "현금",
                "영업활동현금흐름": "영업현금흐름", "투자활동현금흐름": "투자현금흐름", "재무활동현금흐름": "재무현금흐름",
                "영업이익율": "영업이익률", "순이익율": "순이익률"}
BASIS = ["분기", "누적", "연간"]


def _metric(word):
    """지표 이름 → {"name", "grp", "key"} 또는 비율 {"name", "ratio"}"""
    w = METRIC_ALIAS.get(re.sub(r"\s", "", word), re.sub(r"\s", "", word))
    if w in METRICS:
        return {"name": w, "grp": METRICS[w][0], "key": METRICS[w][1]}
    if w in RATIOS:
        return {"name": w, "ratio": RATIOS[w]}
    try:     # 그 밖의 계정은 전 종목 계정 이름표(fs_account)에서
        row = (db().execute("SELECT grp, acc_key, label FROM fs_account WHERE REPLACE(label, ' ', '') = ? "
                            "ORDER BY n_corps DESC LIMIT 1", (w,)).fetchone()
               or db().execute("SELECT grp, acc_key, label FROM fs_account WHERE REPLACE(label, ' ', '') LIKE ? "
                               "ORDER BY n_corps DESC LIMIT 1", (f"%{w}%",)).fetchone())
    except sqlite3.OperationalError:
        row = None
    if row:
        return {"name": row["label"], "grp": row["grp"], "key": row["acc_key"]}
    raise ToolError(f"지표를 모름: {word}. 쓸 수 있는 지표: {', '.join([*METRICS, *RATIOS])}")


def _fin_periods(code, specs):
    """{(연도, 분기): (재무제표 구분, {(grp, key): (누적값, 3개월값)}, 접수일)} — 연결 우선"""
    keys = set()
    for sp in specs:
        for part in ([_metric(x) for x in sp["ratio"]] if "ratio" in sp else [sp]):
            keys.add((part["grp"], part["key"]))
    try:
        rows = db().execute(
            f"SELECT bsns_year, fq, fs_div, grp, acc_key, value, value_q, rcept_date FROM fs_item "
            f"WHERE stock_code = ? AND acc_key IN ({','.join('?' * len(keys))})", [code, *{k for _, k in keys}]).fetchall()
    except sqlite3.OperationalError:
        raise ToolError("재무 데이터(fs_item)가 없습니다 (normalize_financials.py 필요)")
    data, rcpt = {}, {}
    for r in rows:
        if (r["grp"], r["acc_key"]) in keys:
            p = (int(r["bsns_year"]), int(r["fq"]), r["fs_div"])
            data.setdefault(p, {}).setdefault((r["grp"], r["acc_key"]), (r["value"], r["value_q"]))
            rcpt[p] = rcpt.get(p) or r["rcept_date"]
    out = {}
    for (y, q, div), vals in sorted(data.items()):
        if (y, q) not in out or div == "CFS":
            out[(y, q)] = (div, vals, rcpt.get((y, q, div)))
    return dict(sorted(out.items()))


def _val(vals, sp, basis):
    if "ratio" in sp:
        a, b = (_val(vals, _metric(x), basis) for x in sp["ratio"])
        return None if a is None or not b else a / b
    t = vals.get((sp["grp"], sp["key"]))
    if not t:
        return None
    x = t[0] if sp["grp"] == "BS" or basis != "분기" else t[1]
    return None if x is None else float(x)


def _growth(cur, prev):
    if cur is None or prev is None:
        return "없음"
    if prev > 0 and cur > 0:
        return pct(cur / prev - 1)
    if prev <= 0 < cur:
        return "흑자전환"
    if prev > 0 >= cur:
        return "적자전환"
    return "적자지속(축소)" if cur > prev else "적자지속(확대)"


def _fmt(sp, v):
    return pct(v, sign=False) if "ratio" in sp else won(v)


def _prev_q(y, q):
    return (y, q - 1) if q > 1 else (y - 1, 4)


def _fin_table(code, name, specs, periods_expr, basis):
    per = _fin_periods(code, specs)
    if not per:
        raise ToolError(f"{name}의 {', '.join(sp['name'] for sp in specs)} 재무 데이터가 없습니다")
    avail = [p for p in per if basis != "연간" or p[1] == 4]
    s = re.sub(r"\s", "", periods_expr or "")
    m = re.fullmatch(r"(?:최근|지난)?(\d+)(분기|개년|년)", s)
    if m and (s.startswith(("최근", "지난")) or m[2] == "분기"):        # 최근 8분기 / 최근 3년
        n = int(m[1]) if m[2] == "분기" or basis == "연간" else int(m[1]) * 4
        sel = avail[-n:]
    else:
        a, b, _ = parse_period(periods_expr)
        qend = lambda y, q: date(y, 3 * q, monthrange(y, 3 * q)[1]).isoformat()
        sel = [p for p in avail if a <= qend(*p) <= b]
    if not sel:
        have = f"{avail[0][0]}Q{avail[0][1]}~{avail[-1][0]}Q{avail[-1][1]}" if avail else "없음"
        raise ToolError(f"{name}의 {periods_expr} 재무 데이터가 없습니다 (있는 기간: {have})")
    get = lambda p, sp: _val(per[p][1], sp, basis) if p in per else None
    rows = []
    for (y, q) in sel[-20:]:
        div, _, rd = per[(y, q)]
        row = {"기간": f"{y}년" if basis == "연간" else f"{y}Q{q}", "재무제표": "연결" if div == "CFS" else "별도",
               "보고서 접수일": rd or "없음"}
        for sp in specs:
            cur = get((y, q), sp)
            row[sp["name"]] = _fmt(sp, cur)
            if "ratio" not in sp:
                row[f"{sp['name']} 전년 대비"] = _growth(cur, get((y - 1, q), sp))
                if basis == "분기":
                    row[f"{sp['name']} 전분기 대비"] = _growth(cur, get(_prev_q(y, q), sp))
        rows.append(row)
    summary = {}
    last = sel[-1]
    for sp in specs:
        if "ratio" in sp:
            continue
        step = (lambda p: (p[0] - 1, p[1])) if basis != "분기" else (lambda p: _prev_q(*p))
        n, p = 0, last
        while True:          # 최근부터 거슬러 올라가며 연속 증가 횟수
            cur, prv = get(p, sp), get(step(p), sp)
            if cur is None or prv is None or cur <= prv:
                break
            n, p = n + 1, step(p)
        summary[f"{sp['name']} 연속 증가"] = f"{n}{'분기' if basis == '분기' else '년'} 연속 ({'전분기' if basis == '분기' else '전년'} 대비)"
        if basis == "분기" and sp["grp"] != "BS":
            q4, p = [], last
            for _ in range(4):
                q4.append(get(p, sp))
                p = _prev_q(*p)
            if all(v is not None for v in q4):
                summary[f"{sp['name']} 최근 4분기 합"] = won(sum(q4))
    notes = []
    try:
        for y, q, div, chk in db().execute(
                "SELECT bsns_year, fq, fs_div, check_name FROM fs_check WHERE stock_code = ? AND ok = 0", (code,)):
            if (int(y), int(q)) in sel:
                notes.append(f"{y}Q{q} 검증 실패({chk}): 원본 데이터 오류 가능")
    except sqlite3.OperationalError:
        pass
    out = {"종목": f"{name}({code})",
           "기준": {"분기": "분기(3개월)", "누적": "연초부터 누적", "연간": "연간"}[basis] + ", 연결 우선(없으면 별도)",
           "표": rows}
    if summary:
        out["요약"] = summary
    if notes:
        out["주의"] = notes[:5]
    return out


def _reports(code, name, s, e):
    """s~e 사이에 접수된 정기보고서와 그 분기 실적 (전년 대비)"""
    try:
        rows = db().execute("SELECT DISTINCT bsns_year, fq, rcept_date FROM fs_item WHERE stock_code = ? "
                            "AND rcept_date BETWEEN ? AND ? ORDER BY rcept_date", (code, s, e)).fetchall()
    except sqlite3.OperationalError:
        return []
    specs = [_metric(x) for x in ("매출", "영업이익", "순이익")]
    per = _fin_periods(code, specs) if rows else {}
    out = []
    for y, q, rd in rows:
        y, q = int(y), int(q)
        if (y, q) not in per:
            continue
        parts = []
        for sp in specs:
            cur = _val(per[(y, q)][1], sp, "분기")
            prv = _val(per[(y - 1, q)][1], sp, "분기") if (y - 1, q) in per else None
            parts.append(f"{sp['name']} {won(cur)}(전년 대비 {_growth(cur, prv)})")
        out.append(f"{y}Q{q} {'사업' if q == 4 else '분기·반기'}보고서 {rd} 접수: " + ", ".join(parts))
    return list(dict.fromkeys(out))


# ───────────────────────── 뉴스 도우미 ─────────────────────────
def _has_news(code):
    try:
        return db().execute("SELECT COUNT(*) FROM news_cluster WHERE stock_code = ?", (code,)).fetchone()[0] > 0
    except sqlite3.OperationalError:
        return False


def _clu(c):
    s = (f"{c['first_date']} {c['label']} (기사 {c['n_articles']}건, {c['relevance']}·{c['art_type']}·{c['direction']}): "
         f"{(c['summary'] or '')[:150]}")
    return s + (f" / 주가설명: {c['price_note']}" if c.get("price_note") else "")


def _news_range(code):
    """이 종목 뉴스 자료의 첫날·마지막 날 (없으면 None)"""
    try:
        r = db().execute("SELECT MIN(pub_date), MAX(pub_date) FROM news WHERE stock_code = ?", (code,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return (r[0], r[1]) if r and r[0] else None


def _no_news_msg(code, s, e, when):
    rng = _news_range(code)
    if rng and (e < rng[0] or s > rng[1]):
        return f"뉴스 없음: 뉴스 자료 기간({rng[0]}~{rng[1]}) 밖 → 기술적 분석만"
    cover = f" (뉴스 자료 기간 {rng[0]}~{rng[1]})" if rng else ""
    return f"뉴스 없음: {when} 반영된 뉴스가 없음{cover} → 기술적 분석만"


def is_no_news(news):
    return isinstance(news.get("뉴스"), str) and news["뉴스"].startswith("뉴스 없음")


def _news(code, name, s, e, one):
    """뉴스가 없으면 {"뉴스": "뉴스 없음: …"} 으로 분명히 알린다 (→ 기술적 분석만)"""
    if not _has_news(code):
        return {"종목": f"{name}({code})", "뉴스": "뉴스 없음: 이 종목은 뉴스 자료가 없음 → 기술적 분석만"}
    if one:
        r = news_query.day_news(db(), code, s)
        if not any((r["new"], r["recent"], r["cont"], r["threads"], r["price_notes"])):
            d0 = td_shift(r["date"], -news_query.RECENT) if r["date"] in _pos else r["date"]
            out = {"종목": f"{name}({code})", "반영 거래일": r["date"],
                   "뉴스": _no_news_msg(code, d0, r["date"], f"그날과 직전 {news_query.RECENT}거래일에")}
            if r["robot"]:
                out["로봇·시세 기사 (가격 기록일 뿐 원인 아님)"] = r["robot"][:3]
            return out
        return {"종목": f"{name}({code})", "반영 거래일": r["date"],
                "기사 수": f"{r['n_all']}건 (사건 기사 {r['n_event']}건, 직전 60거래일 하루 평균 {r['base60'] or 0:.1f}건)",
                "그날 처음 나온 사건": [_clu(c) for c in r["new"][:5]] or "없음",
                f"직전 {news_query.RECENT}거래일 사건": [_clu(c) for c in r["recent"][:5]] or "없음",
                "더 전에 시작해 그날도 보도된 사건": [_clu(c) for c in r["cont"][:3]] or "없음",
                "이어지는 예전 사건(줄기)": [f"{t['label']}: " + " → ".join(f"{c['first_date']} {c['label']}"
                                                                    for c in t["earlier"][-3:]) for t in r["threads"][:3]] or "없음",
                "로봇·시세 기사": r["robot"][:3] or "없음",
                "주가 이유를 말한 기사": [f"{p['trade_date']} {p['title']} → {p['price_note']}" for p in r["price_notes"][:4]] or "없음"}
    r = news_query.period_news(db(), code, s, e, top=8)
    if not r["n_clusters"]:
        return {"종목": f"{name}({code})", "기간": f"{s}~{e}",
                "뉴스": _no_news_msg(code, s, e, "이 기간에")}
    return {"종목": f"{name}({code})", "기간": f"{s}~{e}", "사건 수": r["n_clusters"],
            "중요 사건": [_clu(c) for c in r["clusters"]] or "없음",
            "월별 요약": [f"{m['month']} ({m['tone'] or '분위기 없음'}): {m['summary']}" for m in r["months"][-6:]] or "없음"}


# ───────────────────────── 도구 ─────────────────────────
STOCK = arg("종목 이름·약칭 또는 6자리 코드", True)
STOCKS = arg("종목 이름 또는 코드. 여러 종목은 쉼표로 (최대 5개)", True)
PERIOD = arg("하루(2026-04-30) 또는 기간(최근 3개월, 2026-01-01~2026-06-30, 2025Q3, 2025년, "
             "2026-06-09부터 1개월, 2026-06-09 이전 1개월)", True)


@tool("종목확인", "이름·약칭·코드로 종목을 찾고, 가격·재무·뉴스 데이터가 있는 기간을 알려 준다. 이름이 애매할 때 쓴다",
      {"이름": arg("종목 이름, 약칭 또는 코드", True)}, {"이름": "하이닉스"})
def 종목확인(이름):
    code, name = resolve(이름)
    c = db()
    p = c.execute("SELECT MIN(date), MAX(date) FROM daily_price WHERE code = ?", (code,)).fetchone()
    out = {"코드": code, "이름": name, "가격": f"{p[0]}~{p[1]}" if p[0] else "없음"}
    try:
        f = c.execute("SELECT MIN(bsns_year * 10 + fq), MAX(bsns_year * 10 + fq), GROUP_CONCAT(DISTINCT fs_div) "
                      "FROM fs_item WHERE stock_code = ?", (code,)).fetchone()
        out["재무"] = (f"{f[0] // 10}Q{f[0] % 10}~{f[1] // 10}Q{f[1] % 10} "
                     f"({'연결' if 'CFS' in (f[2] or '') else ''}{'·별도' if 'OFS' in (f[2] or '') else ''})") if f[0] else "없음"
    except sqlite3.OperationalError:
        out["재무"] = "없음"
    out["뉴스 색인"] = "있음" if _has_news(code) else "없음"
    return out


@tool("가격요약",
      "주가 움직임 숫자. 하루면 등락률·코스피 등락·시장 몫과 고유 등락·거래량 배율·시가 갭·전후 5일 흐름, "
      "기간이면 수익률·코스피 대비·최고최저·최대 낙폭·변동성. 여러 종목은 나란히 비교",
      {"종목": STOCKS, "기간": PERIOD}, {"종목": "KG케미칼", "기간": "2026-04-30"})
def 가격요약(종목, 기간):
    items = resolve_many(종목)
    s, e, one = parse_period(기간)
    outs = [(_day(c, n, s) if one else _period(c, n, s, e))[0] for c, n in items]
    if len(outs) == 1:
        return outs[0]
    res = {"종목별": outs}
    if not one:
        s0, e0 = trading_range(s, e)
        px = prices([c for c, _ in items], s0, e0)
        r = pd.DataFrame({n: px[c]["close"].pct_change() for c, n in items if c in px}).dropna(how="all")
        cm = r.corr()
        res["일별 수익률 상관계수"] = [f"{a}-{b} {cm.loc[a, b]:.2f}" for i, a in enumerate(cm.columns)
                               for b in cm.columns[i + 1:] if not _nan(cm.loc[a, b])]
    return res


@tool("기술지표", "그날 기준 추세·위치: 이동평균(5·20·60·120일) 위아래와 배열, 20일 이격도, RSI(14), "
                 "52주 고가·저가 대비 위치, 신고가 여부, 거래량 추세, 직전 20·60거래일 수익률",
      {"종목": STOCK, "날짜": arg("기준일 (없으면 마지막 거래일)")}, {"종목": "KG케미칼", "날짜": "2026-04-29"})
def 기술지표(종목, 날짜=None):
    code, name = resolve(종목)
    d = parse_period(날짜)[1] if 날짜 else last_day()
    return _tech(code, name, d)


@tool("급등락", "기간 중 코스피 대비 고유 등락이 가장 컸던 날, 또는 20·60거래일 구간",
      {"종목": STOCK, "기간": PERIOD, "방향": arg("상승 / 하락 / 둘다 (기본 둘다)", choices=["상승", "하락", "둘다"]),
       "단위": arg("하루 / 20일 / 60일 (기본 하루)", choices=["하루", "20일", "60일"]),
       "개수": arg("몇 개 (기본 5, 최대 10)", kind="int")},
      {"종목": "KG케미칼", "기간": "최근 1년", "방향": "상승", "단위": "하루", "개수": 5})
def 급등락(종목, 기간, 방향="둘다", 단위="하루", 개수=5):
    code, name = resolve(종목)
    s, e, _ = parse_period(기간)
    s0, e0 = trading_range(s, e)
    n = max(1, min(int(개수 or 5), 10))
    px = prices([code], s0, e0, lookback=260)
    st, kc = series(px, code, name), px["KOSPI"]["close"]
    c = st["close"]
    kc = kc.reindex(c.index)
    idx = list(c.index)
    inwin = [s0 <= d <= e0 for d in idx]
    way = "하락" if re.search(r"하락|내|떨|down", str(방향)) else "상승" if re.search(r"상승|오|급등|up", str(방향)) else "둘다"

    def pick(x):
        x = x.dropna()
        if way == "상승":
            return list(x.sort_values(ascending=False).index)
        if way == "하락":
            return list(x.sort_values().index)
        return list(x.abs().sort_values(ascending=False).index)

    unit = str(단위 or "하루")
    if unit in ("하루", "일", "1일"):
        r, kr = c.pct_change(), kc.pct_change()
        b = beta(r, kr, s0)
        ab = (r - b * kr)[inwin]
        vr = st["volume"] / st["volume"].rolling(20).mean().shift(1)
        rows = [{"날짜": d, "등락률": pct(r[d]), "코스피": pct(kr[d]), "고유 등락": pp(ab[d]),
                 "거래량": "없음" if _nan(vr[d]) else f"{vr[d]:.1f}배"} for d in pick(ab)[:n]]
        basis = f"하루 고유 등락(등락률 - 베타 {b:.2f} x 코스피)"
    else:
        N = int(re.sub(r"\D", "", unit) or 20)
        cum, kcum = c / c.shift(N) - 1, kc / kc.shift(N) - 1
        i_s0 = idx.index(next(d for d in idx if d >= s0))
        valid = [j for j in range(len(idx)) if inwin[j] and j - N >= i_s0 - 1]
        if not valid:
            raise ToolError(f"기간이 {N}거래일보다 짧습니다")
        ex = (cum - kcum).iloc[valid]
        taken = []
        for d in pick(ex):
            j = idx.index(d)
            if all(abs(j - t) >= N for t in taken):
                taken.append(j)
            if len(taken) == n:
                break
        rows = [{"구간": f"{idx[j - N]} 종가 → {idx[j]} 종가", "수익률": pct(cum.iloc[j]), "코스피": pct(kcum.iloc[j]),
                 "코스피 대비": pp(cum.iloc[j] - kcum.iloc[j])} for j in taken]
        basis = f"{N}거래일 구간 수익률 - 코스피 수익률 (겹치지 않게)"
    return {"종목": f"{name}({code})", "기간": f"{s0}~{e0}", "기준": basis, "방향": way, "목록": rows or "해당 없음"}


@tool("시장요약", "하루 또는 기간의 시장 전체: 코스피·코스닥 등락, 코스피 보통주 상승·하락 종목 수, "
                 "상승률·하락률 상위, (하루면) 거래량 급증 종목",
      {"기간": PERIOD}, {"기간": "2026-04-30"})
def 시장요약(기간):
    s, e, one = parse_period(기간)
    if one:
        d1 = td_after(s)
        if d1 is None:
            raise ToolError(f"{s}는 가격 데이터 범위 밖입니다")
        d0 = td_shift(d1, -1)
    else:
        s0, d1 = trading_range(s, e)
        d0 = td_shift(s0, -1) if s0 != calendar()[0] else s0
    c = db()
    idx = {(r[0], r[1]): r[2] for r in c.execute(
        "SELECT code, date, close FROM daily_price WHERE code IN ('KOSPI', 'KOSDAQ') AND date IN (?, ?)", (d0, d1))}
    ir = {k: (idx[(k, d1)] / idx[(k, d0)] - 1 if (k, d0) in idx and (k, d1) in idx else float("nan")) for k in INDEX}
    rows = [tuple(r) for r in c.execute(
        "SELECT p.code, p.date, p.close FROM daily_price p JOIN symbols s ON s.code = p.code "
        "WHERE s.kind = 'stock' AND p.date IN (?, ?) AND p.close > 0", (d0, d1))]
    t = pd.DataFrame(rows, columns=["code", "date", "close"]).pivot(index="code", columns="date", values="close")
    if d0 not in t.columns or d1 not in t.columns:
        raise ToolError("그 기간의 종목 가격이 없습니다")
    r = (t[d1] / t[d0] - 1).dropna()
    nm = names()
    out = {"기간": f"{d0} 종가 → {d1} 종가", "코스피": pct(ir["KOSPI"]), "코스닥": pct(ir["KOSDAQ"]),
           "상승·하락·보합 종목": f"{int((r > 0).sum())} / {int((r < 0).sum())} / {int((r == 0).sum())} (코스피 보통주 {len(r)}개)",
           "등락률 중앙값": pct(r.median()),
           "상승률 상위": ", ".join(f"{nm.get(k, k)} {pct(v)}" for k, v in r.nlargest(5).items()),
           "하락률 상위": ", ".join(f"{nm.get(k, k)} {pct(v)}" for k, v in r.nsmallest(5).items())}
    if one:
        v = pd.DataFrame([tuple(x) for x in c.execute(
            "SELECT p.code, p.date, p.volume FROM daily_price p JOIN symbols s ON s.code = p.code "
            "WHERE s.kind = 'stock' AND p.date BETWEEN ? AND ?", (td_shift(d1, -20), d1))],
            columns=["code", "date", "volume"]).pivot(index="code", columns="date", values="volume")
        if d1 in v.columns and v.shape[1] > 1:
            base = v.drop(columns=d1).mean(axis=1)
            vr = (v[d1] / base.where(base > 0)).dropna().nlargest(5)
            out["거래량 급증"] = ", ".join(f"{nm.get(k, k)} {x:.1f}배({pct(r.get(k, float('nan')))})" for k, x in vr.items())
    return out


@tool("재무조회", "분기 재무 숫자와 증감: 매출·영업이익·순이익·지배순이익·자산·부채·자본·영업현금흐름 등과 "
                 "영업이익률·순이익률·부채비율·유동비율. 전년·전분기 대비, 연속 증가 횟수, 최근 4분기 합, 보고서 접수일",
      {"종목": STOCKS, "지표": arg("지표 이름, 여러 개는 쉼표로 (예: 매출,영업이익,영업이익률)", True),
       "기간": arg("최근 8분기(기본) / 최근 3년 / 2025Q3 / 2024년 / 2023Q1~2025Q4"),
       "기준": arg("분기(3개월, 기본) / 누적 / 연간", choices=BASIS)},
      {"종목": "KG케미칼", "지표": "매출,영업이익,영업이익률", "기간": "최근 8분기", "기준": "분기"})
def 재무조회(종목, 지표, 기간=None, 기준="분기"):
    basis = 기준 if 기준 in BASIS else "분기"
    specs = [_metric(w) for w in re.split(r"[,/]", str(지표 or "")) if w.strip()]
    if not specs:
        raise ToolError("지표가 없습니다")
    expr = 기간 or ("최근 5년" if basis == "연간" else "최근 8분기")
    m = re.fullmatch(r"(\d{4})[Qq]([1-4])~(\d{4})[Qq]([1-4])", re.sub(r"\s", "", expr))
    if m:      # 분기~분기 → 날짜 범위로
        a = date(int(m[1]), 3 * int(m[2]) - 2, 1).isoformat()
        b = date(int(m[3]), 3 * int(m[4]), monthrange(int(m[3]), 3 * int(m[4]))[1]).isoformat()
        expr = f"{a}~{b}"
    outs = [_fin_table(c, n, specs, expr, basis) for c, n in resolve_many(종목)]
    return outs[0] if len(outs) == 1 else {"종목별": outs}


@tool("뉴스요약", "색인된 종목의 뉴스. 하루면 그날 처음 나온 사건·직전 3거래일 사건·계속 보도 중인 사건·이어지는 예전 사건 줄기·"
                 "주가 이유를 말한 기사, 기간이면 중요 사건과 월별 요약",
      {"종목": STOCK, "기간": PERIOD}, {"종목": "KG케미칼", "기간": "2026-04-30"})
def 뉴스요약(종목, 기간):
    code, name = resolve(종목)
    s, e, one = parse_period(기간)
    return _news(code, name, s, e, one)


@tool("뉴스검색", "주제로 뉴스 사건·기사를 찾는다 (키워드 + 의미 검색). 종목·기간은 선택",
      {"주제": arg("찾을 주제 (예: 요소수, 주주환원 발표)", True), "종목": arg("종목 (없으면 색인된 전체)"),
       "기간": arg("기간 (없으면 전체)")}, {"주제": "주주환원 발표", "종목": "KG케미칼"})
def 뉴스검색(주제, 종목=None, 기간=None):
    global _tok
    code, name = resolve(종목) if 종목 else (None, None)
    if code and not _has_news(code):
        return {"주제": 주제, "결과": f"{name}은(는) 뉴스 색인이 없습니다"}
    s, e = parse_period(기간)[:2] if 기간 else (None, None)
    if _tok is None:
        _tok = news_query.Tokenizer(news_query.name_words(db()))
    res = news_query.search(db(), code, 주제, s, e, k=30, tok=_tok)
    nm, out, seen = names(), [], set()
    for r in res:                     # 같은 사건의 기사가 여러 건 걸리면 사건 하나로
        key, day, kind, text = r["doc_id"], r["date"], {"cluster": "사건", "month": "월 요약"}.get(r["doc_type"], "기사"), r["text"]
        if r["doc_type"] == "article":
            sc, nid = r["doc_id"].split(":", 1)
            c = db().execute("SELECT c.cluster_id, c.first_date, c.label, c.summary, c.n_articles FROM news_article a "
                             "JOIN news_cluster c ON c.cluster_id = a.cluster_id WHERE a.stock_code = ? AND a.news_id = ?",
                             (sc, nid)).fetchone()
            if c:
                key, day, kind = c["cluster_id"], c["first_date"], "사건"
                text = f"{c['label']} (기사 {c['n_articles']}건): {c['summary']}"
        elif r["doc_type"] == "cluster":
            text = text.split(" | ")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{day} [{nm.get(r['stock_code'], r['stock_code'])}] {kind}: {text[:180]}")
        if len(out) == 8:
            break
    return {"주제": 주제, "결과": out or "관련 뉴스 없음 (뉴스 색인은 일부 종목만 있음)"}


@tool("움직임설명", "\"왜 올랐어/내렸어\" 질문용 묶음 도구. 가격 움직임을 시장 몫과 고유 몫으로 나누고, 움직임 전 기술적 상태, "
                   "그날(기간)의 기술적 신호, 그날 시장, 뉴스(새 사건·직전 사건·이어진 줄기), 그 기간 실적 보고서, 판단 참고를 "
                   "한 번에 돌려준다. 뉴스가 없으면 '뉴스 없음'과 기술적 분석 자료만 돌려준다",
      {"종목": STOCK, "기간": PERIOD}, {"종목": "KG케미칼", "기간": "2026-04-30"})
def 움직임설명(종목, 기간):
    code, name = resolve(종목)
    s, e, one = parse_period(기간)
    hints = []
    if one:
        price, raw = _day(code, name, s)
        d0 = raw["date"]
        tech = _tech(code, name, td_shift(d0, -1))
        market = 시장요약(d0)
        news = _news(code, name, d0, d0, True)
        reps = _reports(code, name, td_shift(d0, -5), d0)
        moves = None
        signals = ("그날 기술적 신호", _day_signals(code, name, d0))
    else:
        price, raw = _period(code, name, s, e)
        s0, e0 = raw["s0"], raw["e0"]
        tech = _tech(code, name, td_shift(s0, -1))
        market = 시장요약(f"{s0}~{e0}")
        news = _news(code, name, s0, e0, False)
        reps = _reports(code, name, s0, e0)
        moves = 급등락(code, f"{s0}~{e0}", "둘다", "하루", 3)["목록"]
        signals = ("기간 기술적 변화", _period_signals(code, name, s0, e0))
    no_news = is_no_news(news)
    ret, own = raw["ret"], raw["own"]
    if not _nan(own):
        if abs(own) < 0.01 or abs(own) < 0.3 * abs(ret):
            hints.append(f"대부분 시장 흐름: 고유 등락 {pp(own)} (전체 {pct(ret)})")
        else:
            hints.append(f"종목 고유 움직임이 큼: 고유 등락 {pp(own)} (전체 {pct(ret)}, 코스피 {pct(raw['kret'])})")
    vr = raw["vol_ratio"]
    if not _nan(vr):
        hints.append(f"거래량 {'급증' if vr >= 2 else '적음' if vr <= 0.7 else '보통'} ({vr:.1f}배)")
    if no_news:
        hints.append("뉴스 없음: 뉴스 원인을 말하지 말고 시장 흐름·거래량·기술적 신호로만 설명")
    elif one:
        cnt = {k: (len(news[k]) if isinstance(news[k], list) else 0)
               for k in ("그날 처음 나온 사건", f"직전 {news_query.RECENT}거래일 사건", "이어지는 예전 사건(줄기)", "주가 이유를 말한 기사")}
        hints.append("뉴스: " + ", ".join(f"{k} {v}개" for k, v in cnt.items()))
    hints.append(f"실적 보고서 {len(reps)}건 접수" if reps else "이 기간 정기보고서 접수 없음 (잠정실적 공시는 데이터에 없음)")
    keep_t = ("배열", "20일 이격도", "RSI(14)", "52주 범위 안 위치", "신고가", "직전 20거래일 수익률", "직전 60거래일 수익률")
    out = {"가격": price,
           "시장": {k: market[k] for k in ("기간", "코스피", "코스닥", "상승·하락·보합 종목") if k in market},
           "움직임 전 기술적 상태": tech if no_news else {"기준일": tech["기준일"], **{k: tech[k] for k in keep_t}},
           signals[0]: signals[1], "뉴스": news, "실적 보고서": reps or "없음", "판단 참고": hints}
    if moves is not None:
        out["기간 중 큰 움직임"] = moves
    return out


# ───────────────────────── 혼자 시험 ─────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print(__doc__)
        for n, t in TOOLS.items():
            print(f"  {n}({', '.join(t['args'])}): {t['desc'][:60]}")
        sys.exit()
    kw = dict(a.split("=", 1) for a in sys.argv[2:])
    try:
        print(json.dumps(TOOLS[sys.argv[1]]["fn"](**kw), ensure_ascii=False, indent=2))
    except ToolError as err:
        print(f"오류: {err}")
