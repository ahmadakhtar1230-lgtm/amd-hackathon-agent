import json
import os
import sys
import time
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
API_KEY = os.environ.get("FIREWORKS_API_KEY")
RAW_BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS_RAW = os.environ.get("ALLOWED_MODELS")

INPUT_PATH = "/input/tasks.json"
OUTPUT_DIR = "/output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results.json")
LOCAL_FALLBACK_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tasks.json")

# ============================================================
# OLLAMA (LOCAL) CONFIG
# ============================================================
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LOCAL_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 60
REQUEST_TIMEOUT_SECONDS = 30
MAX_WORKERS = 4 

PREFERRED_MODEL_SUBSTRINGS = [
    "minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it-nvfp4",
    "gemma-4-31b-it", "gemma-4-26b-a4b-it",
]

# ============================================================
# PROMPTS
# ============================================================
FW_BASE_RULE = (
    "You are a highly concise precision AI. Output ONLY the exact final answer or code. "
    "BE EXTREMELY CONCISE. Zero preamble, zero explanations. Just the final answer."
)
LOCAL_BASE_RULE = "You are a helpful AI assistant. Please provide the correct answer."

CATEGORY_PROMPTS = {
    "factual": "Provide just the facts.",
    "math": "Output ONLY the final numerical answer.",
    "sentiment": "Output ONLY one word: Positive, Negative, or Neutral.",
    "summarization": "Follow the length constraint and summarize.",
    "ner": "Output ONLY compact JSON: {\"PERSON\":[],\"ORG\":[],\"LOCATION\":[],\"DATE\":[]}.",
    "code_debug": "Output ONLY the corrected code in a single code block. NO explanation.",
    "logic": "Output ONLY the final answer. Extremely concise.",
    "code_gen": "Output ONLY the working code in a single code block. NO comments, NO descriptions.",
}

# ============================================================
# SMART ROUTER & FIXED THINKING BUDGETS
# ============================================================
HARD_CATEGORIES = {"math", "logic", "code_debug", "code_gen", "ner"}

# Minimax STRICTLY requires >=1024 for budget_tokens. 
# Code is excluded from reasoning to save tokens and prevent cut-offs.
REASONING_ENABLED_FOR = {"math", "logic"} 
FALLBACK_THINKING_BUDGET = 1024 

MAX_TOKENS_BY_KEY = {
    "factual": 200, "math": 200, "sentiment": 20, "summarization": 300,
    "ner": 250, "code_debug": 2000, "logic": 200, "code_gen": 2000,
}

def log(msg):
    print(msg, file=sys.stderr, flush=True)

def fail(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)

def start_ollama():
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log(f"Could not start ollama serve: {e}")
        return False
    for _ in range(30):
        try:
            if requests.get("http://127.0.0.1:11434/api/tags", timeout=3).status_code == 200:
                log("Ollama is up.")
                return True
        except Exception:
            pass
        time.sleep(1)
    log("Ollama did not come up in time.")
    return False

def classify(task):
    category = str(task.get("category", "")).lower()
    prompt = str(task.get("prompt", "")).lower()

    if "math" in category: return "math"
    elif "sentiment" in category: return "sentiment"
    elif "summar" in category: return "summarization"
    elif "entity" in category or "ner" in category: return "ner"
    elif "debug" in category: return "code_debug"
    elif "generat" in category: return "code_gen"
    elif "logic" in category or "deduct" in category: return "logic"
    elif "factual" in category or "knowledge" in category: return "factual"
    
    if any(k in prompt for k in ["calculate", "how many", "percent", "sum of", "%"]): return "math"
    if any(k in prompt for k in ["sentiment", "positive or negative"]): return "sentiment"
    if "summar" in prompt: return "summarization"
    if any(k in prompt for k in ["extract", "entities", "named entity"]): return "ner"
    if any(k in prompt for k in ["bug", "fix this code", "debug", "error in"]): return "code_debug"
    if any(k in prompt for k in ["write a function", "implement", "def ", "write code"]): return "code_gen"
    if any(k in prompt for k in ["puzzle", "constraint", "deduce"]): return "logic"
    return "factual"

def ask_local(system_prompt, prompt, key):
    payload = {
        "model": LOCAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            ans = (r.json().get("message", {}).get("content") or "").strip()
            if ans: return ans
    except Exception as e:
        log(f"Ollama error: {e}")
    return None

def resolve_chat_url(base_url):
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"

def pick_model(allowed_list):
    for pref in PREFERRED_MODEL_SUBSTRINGS:
        for m in allowed_list:
            if pref in m: return m
    return allowed_list[0] if allowed_list else None

def extract_api_answer(data):
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if content: return content
    reasoning = (msg.get("reasoning_content") or "").strip()
    if reasoning:
        lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
        if lines: return lines[-1]
    return ""

def ask_fireworks(system_prompt, prompt, key, chat_url, headers, model_name, is_minimax):
    rescue = key in REASONING_ENABLED_FOR
    base_ceiling = MAX_TOKENS_BY_KEY.get(key, 500)
    
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
    
    if is_minimax:
        payload["thinking"] = {"type": "enabled", "budget_tokens": FALLBACK_THINKING_BUDGET} if rescue else {"type": "disabled"}

    try:
        resp = requests.post(chat_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            data = resp.json()
            ans = extract_api_answer(data)
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return ans, tokens
        else:
            log(f"Fireworks HTTP {resp.status_code}")
    except Exception as e:
        log(f"Fireworks error: {e}")
    return "", 0

def process_task(task, fw_ctx):
    task_id = task.get("task_id", task.get("id", "unknown"))
    prompt = task.get("prompt", "")
    key = classify(task)

    fw_system_prompt = FW_BASE_RULE + " " + CATEGORY_PROMPTS[key]
    local_system_prompt = LOCAL_BASE_RULE + " " + CATEGORY_PROMPTS[key]

    # ==== SMART ROUTER ====
    if key in HARD_CATEGORIES and fw_ctx is not None:
        ans, tokens = ask_fireworks(fw_system_prompt, prompt, key, fw_ctx["chat_url"], fw_ctx["headers"], fw_ctx["model_name"], fw_ctx["is_minimax"])
        if ans:
            log(f"Task {task_id} answered FIREWORKS [{key}] tokens={tokens}")
            return {"task_id": task_id, "answer": ans}, tokens

    # Easy categories or if Fireworks failed (Local Fallback)
    ans = ask_local(local_system_prompt, prompt, key)
    if ans:
        log(f"Task {task_id} answered LOCAL [{key}] tokens=0")
        return {"task_id": task_id, "answer": ans}, 0

    log(f"Task {task_id} FAILED completely")
    return {"task_id": task_id, "answer": ""}, 0

def load_tasks():
    path = INPUT_PATH if os.path.isfile(INPUT_PATH) else LOCAL_FALLBACK_INPUT
    if not os.path.isfile(path): fail(f"No input file found at {INPUT_PATH}")
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        fail(f"Could not parse input JSON: {e}")

def write_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f: json.dump(results, f, indent=2, ensure_ascii=False)

def run_agent():
    local_ok = start_ollama()
    if not local_ok: log("WARNING: Ollama not available.")

    fw_ctx = None
    if API_KEY and RAW_BASE_URL and ALLOWED_MODELS_RAW:
        allowed_list = [m.strip() for m in ALLOWED_MODELS_RAW.split(",") if m.strip()]
        model_name = pick_model(allowed_list)
        if model_name:
            fw_ctx = {
                "chat_url": resolve_chat_url(RAW_BASE_URL),
                "headers": {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                "model_name": model_name,
                "is_minimax": "minimax-m3" in model_name,
            }
            log(f"Fireworks ready with model: {model_name}")

    tasks = load_tasks()
    results = []
    total_tokens = 0

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_task, task, fw_ctx) for task in tasks]
            for future in as_completed(futures):
                result, tokens = future.result()
                results.append(result)
                total_tokens += tokens
    finally:
        write_results(results)

    log(f"DONE. total_tasks={len(results)} total_tokens={total_tokens}")
    sys.exit(0)

if __name__ == "__main__":
    run_agent()
