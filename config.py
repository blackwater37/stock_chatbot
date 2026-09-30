# -*- coding: utf-8 -*-


# ─────────────────────────────────────────────
# 채팅 모델 서버
# ─────────────────────────────────────────────
#  소설 작업용으로 띄우던 서버를 그대로 써도 된다. 뉴스 색인 요청은 짧다(3천 토큰 이하).
#  llama-server.exe ^
#    -m gemma-4-26B-A4B-it-QAT\gemma-4-26B-A4B-it-QAT-Q4_0.gguf ^
#    -ngl 999 -fa on -fit off ^
#    -c 49152 --parallel 1 --port 1234 ^
#    --reasoning off ^
#    --dry-multiplier 0
LM_STUDIO_URL = "http://localhost:1234/v1"
API_KEY = "not-needed"
WRITER_MODEL = "gemma-4-26B-A4B-it-QAT-Q4_0"   # 로그·캐시 구분용: 실제 로드한 모델명과 맞춰 둔다

# ─────────────────────────────────────────────
# 임베딩 모델 서버 (별도 포트)
# ─────────────────────────────────────────────
#  llama-server.exe -m bge-m3-Q8_0.gguf --embedding --pooling cls -ngl 999 ^
#    -c 8192 -b 2048 -ub 2048 --port 1235
#  -ub(물리 배치)가 문장 길이보다 작으면 임베딩 요청이 실패한다.
EMBED_URL = "http://localhost:1235/v1"
EMBED_MODEL = "bge-m3"

# ─────────────────────────────────────────────
# 토큰
# ─────────────────────────────────────────────
CONTEXT_LENGTH = 16384
MAX_OUTPUT_TOKENS = 2000
CHARS_PER_TOKEN = 1.33      # 한글 실측: 1토큰당 약 1.33자
