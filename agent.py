import json
import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# ENVIRONMENT VARIABLES — harness injects these at evaluation
# time. NEVER hardcode the API key, base URL, or model name.
# ============================================================
API_KEY = os.environ.get("FIREWORKS_API_KEY")
RAW_BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS_RAW = os.environ.get("ALLOWED_MODELS")

INPUT_PATH = "/input/tasks.json"
OUTPUT_DIR = "/output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")

# Local-dev fallback ONLY (used if /input/tasks.json isn't present,
# e.g. testing on your own machine before the harness is wired up)
LOCAL_FALLBACK_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tasks.json")

REQUEST_TIMEOUT_SECONDS = 25          # must stay under the 30s per-request rule
MAX_WORKERS = 6                       # parallel requests so we finish inside the 10-minute limit

# ------------------------------------------------------------
# TOKEN-MINIMIZATION MODE
# Thinking is OFF for every category by default — this was the single
# biggest token cost (was eating 1000+ tokens per hard task alone).
#
# If accuracy ever drops below your floor, put ONLY the struggling
# category name(s) in here, e.g. {"math"} or {"math", "logic"} — this
# re-enables a SMALL reasoning budget for just those categories, as a
# targeted rescue, instead of turning thinking back on everywhere.
# ------------------------------------------------------------
REASONING_ENABLED_FOR = set()          # e.g. {"math", "logic"} — empty = thinking OFF for everything
FALLBACK_THINKING_BUDGET = 512         # small budget, only used for categories listed above

# Generous-but-not-wasteful output ceilings per category. These are safety
# ceilings, not targets — raising max_tokens does NOT use more tokens by
# itself, it only prevents truncation/errors on longer answers (code_gen).
MAX_TOKENS_BY_KEY = {
    "factual": 100,
    "math": 60,
    "sentiment": 60,
    "summarization": 220,
    "ner": 180,
    "code_debug": 400,
    "logic": 90,
    "code_gen": 550,
}

BASE_RULE = (
    "You are a precision answer engine. Respond only in English. Give the correct "
    "answer in the minimum number of tokens. Output ONLY the final answer. No "
    "preamble, no greetings, no phrases like 'Sure' or 'The answer is'. No markdown "
    "unless it is code."
)

CATEGORY_PROMPTS = {
    "factual": "Answer with ONLY the fact. Fewest words possible. No full sentence needed.",
    "math": "Output ONLY the final numerical answer. Nothing else.",
    "sentiment": "Output ONLY one word (Positive, Negative, or Neutral) followed by a one-sentence justification.",
    "summarization": "Follow the exact length/format constraint in the prompt. Summary only.",
    "ner": "Output ONLY compact JSON: {\"PERSON\":[],\"ORG\":[],\"LOCATION\":[],\"DATE\":[]}. No prose.",
    "code_debug": "Output ONLY the corrected code in one code block. No comments, no explanation.",
    "logic": "Output ONLY the final answer that satisfies all constraints. Fewest words.",
    "code_gen": "Output ONLY the working function in one code block. No comments, no docstrings.",
}

# Preference order for picking a model out of whatever ALLOWED_MODELS contains.
# 'thinking' control is only applied when the chosen model is MiniMax M3 —
# other models don't support that parameter and will error if you send it.
PREFERRED_MODEL_SUBSTRINGS = [
    "minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it-nvfp4",
    "gemma-4-31b-it", "gemma-4-26b-a4b-it",
]


def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def resolve_chat_url(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def pick_model(allowed_list):
    for pref in PREFERRED_MODEL_SUBSTRINGS:
        for m in allowed_list:
            if pref in m:
                return m
    return allowed_list[0] if allowed_list else None


def classify(task):
    category = str(task.get("category", "")).lower()
    prompt = str(task.get("prompt", "")).lower()

    if "math" in category:                                    key = "math"
    elif "sentiment" in category:                             key = "sentiment"
    elif "summar" in category:                                key = "summarization"
    elif "entity" in category or "ner" in category:           key = "ner"
    elif "debug" in category:                                 key = "code_debug"
    elif "generat" in category:                               key = "code_gen"
    elif "logic" in category or "deduct" in category:         key = "logic"
    elif "factual" in category or "knowledge" in category:    key = "factual"
    elif any(k in prompt for k in ["calculate", "how many", "percent", "sum of", "%"]):
        key = "math"
    elif any(k in prompt for k in ["sentiment", "positive or negative"]):
        key = "sentiment"
    elif "summar" in prompt:
        key = "summarization"
    elif any(k in prompt for k in ["extract", "entities", "named entity"]):
        key = "ner"
    elif any(k in prompt for k in ["bug", "fix this code", "debug", "error in"]):
        key = "code_debug"
    elif any(k in prompt for k in ["write a function", "implement", "def ", "write code"]):
        key = "code_gen"
    elif any(k in prompt for k in ["puzzle", "constraint", "deduce", "who is", "which one"]):
        key = "logic"
    else:
        key = "factual"

    return BASE_RULE + " " + CATEGORY_PROMPTS[key], key


def extract_answer(data):
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if content:
        return content
    reasoning = (msg.get("reasoning_content") or "").strip()
    if reasoning:
        lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
        if lines:
            return lines[-1]
    return ""


def load_tasks():
    path = INPUT_PATH if os.path.isfile(INPUT_PATH) else LOCAL_FALLBACK_INPUT
    if not os.path.isfile(path):
        fail(f"No input file found at {INPUT_PATH} (and no local fallback at {LOCAL_FALLBACK_INPUT})")
    try:
        with open(path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if path == LOCAL_FALLBACK_INPUT:
            print(f"NOTE: using local fallback input file ({path}), not {INPUT_PATH}", file=sys.stderr)
        return tasks
    except Exception as e:
        fail(f"Could not parse input JSON at {path}: {e}")


def process_task(task, chat_url, headers, model_name, is_minimax):
    task_id = task.get("task_id", task.get("id", "unknown"))
    prompt = task.get("prompt", "")
    system_prompt, key = classify(task)

    rescue = key in REASONING_ENABLED_FOR   # True only for categories you explicitly listed
    base_ceiling = MAX_TOKENS_BY_KEY.get(key, 300)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1,
        "max_tokens": (FALLBACK_THINKING_BUDGET + base_ceiling) if rescue else base_ceiling,
    }

    # 'thinking' control only exists for MiniMax M3 - sending it to any
    # other model causes an "unsupported parameter" error.
    if is_minimax:
        payload["thinking"] = (
            {"type": "enabled", "budget_tokens": FALLBACK_THINKING_BUDGET}
            if rescue else {"type": "disabled"}
        )

    tokens_used = 0
    answer = ""
    try:
        resp = requests.post(chat_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            answer = extract_answer(data)
            tokens_used = data.get("usage", {}).get("total_tokens", 0)
            tag = "RESCUE" if rescue else key
            print(f"Task {task_id} OK [{tag}] tokens={tokens_used}", file=sys.stderr)
        else:
            print(f"Task {task_id} HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
    except requests.exceptions.RequestException as e:
        print(f"Task {task_id} request error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Task {task_id} unexpected error: {e}", file=sys.stderr)

    return {"task_id": task_id, "answer": answer}, tokens_used


def write_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_agent():
    if not API_KEY:
        fail("FIREWORKS_API_KEY not set in environment")
    if not RAW_BASE_URL:
        fail("FIREWORKS_BASE_URL not set in environment")
    if not ALLOWED_MODELS_RAW:
        fail("ALLOWED_MODELS not set in environment")

    allowed_list = [m.strip() for m in ALLOWED_MODELS_RAW.split(",") if m.strip()]
    model_name = pick_model(allowed_list)
    if not model_name:
        fail("Could not pick a model from ALLOWED_MODELS")

    is_minimax = "minimax-m3" in model_name
    chat_url = resolve_chat_url(RAW_BASE_URL)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    print(f"Using model: {model_name}", file=sys.stderr)
    print(f"Using endpoint: {chat_url}", file=sys.stderr)

    tasks = load_tasks()
    results = []
    total_tokens = 0

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_task, task, chat_url, headers, model_name, is_minimax)
                for task in tasks
            ]
            for future in as_completed(futures):
                result, tokens = future.result()
                results.append(result)
                total_tokens += tokens
    finally:
        # ALWAYS write whatever we have, even on a crash/timeout,
        # so we never score OUTPUT_MISSING.
        write_results(results)

    print(f"DONE. total_tasks={len(results)} total_tokens={total_tokens}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    run_agent()
