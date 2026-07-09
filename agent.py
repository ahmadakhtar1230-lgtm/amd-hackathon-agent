import json
import requests
import os

MY_API_KEY = os.environ.get("FIREWORKS_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/models/minimax-m3"

BASE_RULE = (
    "You are a precision answer engine. Give the correct answer in the minimum "
    "number of tokens. Output ONLY the final answer. No preamble, no greetings, "
    "no phrases like 'Sure' or 'The answer is'. No markdown unless it is code."
)

CATEGORY_PROMPTS = {
    "factual": "Answer with ONLY the fact. Fewest words possible. No full sentence needed.",
    "math": "Output ONLY the final numerical answer. Nothing else.",
    "sentiment": "Output ONLY one word: Positive, Negative, or Neutral.",
    "summarization": "Follow the exact length/format constraint in the prompt. Summary only.",
    "ner": "Output ONLY compact JSON: {\"PERSON\":[],\"ORG\":[],\"LOCATION\":[],\"DATE\":[]}. No prose.",
    "code_debug": "Output ONLY the corrected code in one code block. No comments, no explanation.",
    "logic": "Output ONLY the final answer that satisfies all constraints. Fewest words.",
    "code_gen": "Output ONLY the working function in one code block. No comments, no docstrings.",
}

# Categories jinko sochne ki zaroorat hai -> thinking ENABLED (with a tight budget)
HARD_CATEGORIES = ["math", "logic", "deduct", "debug", "generat", "code"]

# Hard task ke liye thinking budget (>=1024 required by API). Chhota rakha hai
# taaki reasoning explode na kare jaisa Task 7 mein hua tha.
HARD_THINKING_BUDGET = 1536


def classify(task):
    """Returns (extra_prompt, category_key)."""
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
    else:
        key = "factual"

    return BASE_RULE + " " + CATEGORY_PROMPTS[key], key


def needs_reasoning(category_field, key):
    """Hard tasks -> thinking enabled. Easy tasks -> thinking disabled."""
    cat = str(category_field).lower()
    if any(k in cat for k in HARD_CATEGORIES):
        return True
    if key in ("math", "logic", "code_debug", "code_gen"):
        return True
    return False


def extract_answer(data):
    msg = data['choices'][0]['message']
    content = (msg.get('content') or "").strip()
    if content:
        return content, "content"
    reasoning = (msg.get('reasoning_content') or "").strip()
    if reasoning:
        lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
        if lines:
            return lines[-1], "reasoning_fallback"
    return "", "empty"


def run_agent():
    if not MY_API_KEY:
        print("Error: FIREWORKS_API_KEY environment variable set nahi hai!")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_path = os.path.join(script_dir, "test_tasks.json")

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Error: test_tasks.json nahi mili - {e}")
        return

    headers = {"Authorization": f"Bearer {MY_API_KEY}", "Content-Type": "application/json"}
    results = []
    total_tokens = 0

    for i, task in enumerate(tasks, 1):
        system_prompt, key = classify(task)
        hard = needs_reasoning(task.get("category", ""), key)

        if hard:
            thinking_cfg = {"type": "enabled", "budget_tokens": HARD_THINKING_BUDGET}
            max_tokens = HARD_THINKING_BUDGET + 400  # budget + room for the final answer
        else:
            thinking_cfg = {"type": "disabled"}
            max_tokens = 300  # easy tasks: short answer, no thinking phase at all

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.get("prompt", "")}
            ],
            "temperature": 0.0,
            "top_p": 1,
            "thinking": thinking_cfg,   # <-- correct M3 control, NOT reasoning_effort
            "max_tokens": max_tokens,
        }

        ans = ""
        try:
            response = requests.post(BASE_URL, json=payload, headers=headers, timeout=180)

            if response.status_code == 200:
                data = response.json()
                finish = data['choices'][0].get('finish_reason', '')
                ans, source = extract_answer(data)

                if not ans:
                    ans = "ERROR: empty content. Raw: " + json.dumps(data)
                    print(f"Task {i}: EMPTY (finish={finish})")
                else:
                    used = data.get("usage", {}).get("total_tokens", 0)
                    total_tokens += used
                    tag = "HARD" if hard else "easy"
                    flag = "" if source == "content" else f" [{source}]"
                    warn = "  <-- LENGTH LIMIT! (raise HARD_THINKING_BUDGET)" if finish == "length" else ""
                    print(f"Task {i} OK [{tag}] | Tokens: {used}{flag}{warn}")
            else:
                ans = f"Error {response.status_code}: {response.text}"
                print(ans)

        except Exception as e:
            ans = str(e)
            print(f"Task {i} System Error: {ans}")

        results.append({"id": task.get('id', i), "answer": ans})

    results_path = os.path.join(script_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("========================================")
    print(f"DONE. TOTAL TOKENS USED: {total_tokens}")
    print("========================================")


if __name__ == "__main__":
    run_agent()
