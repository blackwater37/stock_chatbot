# -*- coding: utf-8 -*-


import re

import requests

import config

_THINK = [r'<\|think\|>.*?<\|/think\|>', r'<think>.*?</think>',
          r'<\|channel\|?>\s*thought', r'<channel\|?>']


def chat(system, user, max_tokens=None, strip_think=True,
         return_truncated=False, **샘플러):
    """시스템/유저 프롬프트로 호출한다.

    return_truncated=False(기본): 응답 텍스트(str)만 반환.
    return_truncated=True: (텍스트, 잘림여부:bool). 잘림여부는
      finish_reason == 'length'(max_tokens 상한에 걸려 끊김)일 때 True.
    """
    if max_tokens is None:
        max_tokens = config.MAX_OUTPUT_TOKENS

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": config.WRITER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    #  호출부가 준 것만 싣는다. 안 준 것은 서버 기본값이 쓰인다.
    payload.update({k: v for k, v in 샘플러.items() if v is not None})

    try:
        resp = requests.post(f"{config.LM_STUDIO_URL}/chat/completions",
                             json=payload, timeout=1800)
        if resp.status_code != 200:
            raise RuntimeError(f"서버 {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        finish = choice.get("finish_reason", "")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "llama-server 에 연결 불가. 서버를 띄웠는지, 포트가 맞는지 확인.")

    truncated = (finish == "length")
    if truncated:
        print(f"   ⚠ 출력이 max_tokens({max_tokens})에 걸려 잘렸습니다 "
              f"— max_tokens 를 늘리거나 출력을 줄이세요.")
    if strip_think:
        for p in _THINK:
            text = re.sub(p, '', text, flags=re.DOTALL)
    text = text.strip()
    if return_truncated:
        return text, truncated
    return text


def embed(texts, timeout=600):
    """임베딩 서버에 문장 목록을 보내고 같은 순서의 벡터 목록을 받는다."""
    try:
        resp = requests.post(f"{config.EMBED_URL}/embeddings",
                             json={"model": config.EMBED_MODEL, "input": list(texts)},
                             timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "임베딩 서버에 연결 불가. llama-server --embedding 을 EMBED_URL 포트로 띄웠는지 확인.")
    if resp.status_code != 200:
        raise RuntimeError(f"임베딩 서버 {resp.status_code}: {resp.text[:400]}")
    data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def estimate_tokens(text):
    """대강의 토큰 수. CHARS_PER_TOKEN 은 실측값이다(config 주석 참고)."""
    return int(len(text) / config.CHARS_PER_TOKEN)
