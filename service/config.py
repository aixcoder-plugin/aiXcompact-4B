import os

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "aiXcompact")
LLM_TOKEN = os.getenv("LLM_TOKEN", "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "32768"))
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "0").lower() in ("1", "true", "yes")

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "2000"))
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))

# Rough chars-per-token ratio for estimation
CHARS_PER_TOKEN = 3.5
TOKEN_WARN_THRESHOLD = int(os.getenv("TOKEN_WARN_THRESHOLD", "100000000"))

# Target compression ratio (percent of input tokens): injected into the
# system prompt and used as the deterministic token budget for the fallback.
# The ONLY ratio knob — the valid retention band is derived from it as
# [target - RETENTION_BAND_PCT, target + RETENTION_BAND_PCT] percentage
# points, clamped to [0, 100]. retention = output_tokens / input_tokens;
# a result outside the band discards the model's decisions and rebuilds
# deterministically to the target budget.
TARGET_RATIO_PCT = int(os.getenv("TARGET_RATIO_PCT", "25"))
# Fixed band half-width in percentage points. Deliberately a plain constant:
# NOT configurable via env or request — the band is a service invariant,
# not a knob.
RETENTION_BAND_PCT = 20
# Inputs below this many tokens skip the lower-bound check: a tiny trajectory
# can legitimately compress to almost nothing without losing anything
# important.
FLOOR_SKIP_INPUT_TOKENS = int(os.getenv("FLOOR_SKIP_INPUT_TOKENS", "2000"))

# Optional trace retention. Tracing itself is disabled unless TRACE_ENABLED=1.
TRACE_MAX_DIRS = int(os.getenv("TRACE_MAX_DIRS", "0"))

if not 1 <= TARGET_RATIO_PCT <= 99:
    raise ValueError("TARGET_RATIO_PCT must be in [1, 99]")
