import json
import os
import sys
import requests
import time
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

REQUEST_TIMEOUT_SECONDS = 25
MAX_WORKERS = 6

# ============================================================
# ACCURACY BOOST MODE
# Reasoning is ON for hard categories. This increases accuracy 
# heavily by giving the model space to think before answering.
# ============================================================
REASONING_ENABLED_FOR = {"math", "logic", "code_debug", "code_gen"}
FALLBACK_THINKING_BUDGET = 1024  # Increased for better accuracy

MAX_TOKENS_BY_KEY = {
    "factual": 200,
    "math": 150,
    "sentiment": 20,       # Reduced to force exactly 1 word
    "summarization": 300,
    "ner": 250,
    "code_debug": 1024,    # Increased to prevent code truncation
    "logic": 200,
    "code_gen": 1024,      # Increased to prevent code truncation
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
    "sentiment": "Output ONLY one word (Positive, Negative, or Neutral). Nothing else.", # Fixed bug
    "summarization": "Follow the exact length/format constraint in the prompt. Summary only.",
    "ner": "Output ONLY compact JSON: {\"PERSON\":[],\"ORG\":[],\"LOCATION\":[],\"DATE\":[]}. No prose.",
    "code_debug": "Output ONLY the corrected code in one code block. No comments, no explanation.",
    "logic": "Output ONLY the final answer that satisfies all constraints. Fewest words.",
    "code_gen": "Output ONLY the working function in one code block. No comments, no docstrings.",
}

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

    if "math" in category:                                  key = "math"
    elif "sentiment" in category:                           key = "sentiment"
    elif "summar" in category:                              key = "summarization"
    elif "entity" in category or "ner" in category:         key = "ner"
    elif "debug" in category:                               key = "code_debug"
    elif "generat" in category:                             key = "code_gen"
    elif "logic" in category or "deduct" in category:       key = "logic"
    elif "factual" in category or "knowledge" in category:  key = "factual"
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
    elif any(k in prompt for k in ["puzzle", "constraint", "deduce"]): # Fixed factual misclassification
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

    rescue = key in REASONING_ENABLED_FOR   
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

    if is_minimax:
        payload["thinking"] = (
            {"type": "enabled", "budget_tokens": FALLBACK_THINKING_BUDGET}
            if rescue else {"type": "disabled"}
        )

    tokens_used = 0
    answer = ""
    
    # Retry Logic Added: Tries 2 times if it gets an empty answer or error
    for attempt in range(2):
        try:
            resp = requests.post(chat_url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                answer = extract_answer(data)
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                tag = "REASONING_ON" if rescue else key
                print(f"Task {task_id} OK [{tag}] tokens={tokens_used}", file=sys.stderr)
                if answer: 
                    break # Success, exit retry loop
            else:
                print(f"Task {task_id} HTTP {resp.status_code} on attempt {attempt+1}", file=sys.stderr)
        except Exception as e:
            print(f"Task {task_id} error on attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(1) # Wait 1 second before retry

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
        write_results(results)

    print(f"DONE. total_tasks={len(results)} total_tokens={total_tokens}", file=sys.stderr)
    sys.exit(0)

if __name__ == "__main__":
    run_agent()
