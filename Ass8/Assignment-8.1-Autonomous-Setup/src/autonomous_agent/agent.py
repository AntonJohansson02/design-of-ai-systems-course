from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from ollama import Client, ResponseError, chat
except ImportError as exc:
    raise SystemExit("Missing dependency 'ollama'. Install with: pip install ollama") from exc

MODEL = "qwen3.5:2b"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "prepared"
TARGET_FILE = PACKAGE_DIR / "train.py"
BACKUP_FILE = PROJECT_ROOT / "outputs" / "train_backup.py"
RESULTS_FILE = PROJECT_ROOT / "outputs" / "results.tsv"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def ensure_results_header() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        return
    header = ["timestamp_utc", "iteration", "score", "status", "retries", "model", "temperature", "code_hash", "note"]
    with RESULTS_FILE.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(header)


def append_result(iteration: int, score: float | None, status: str, retries: int, model: str, temperature: float, code: str, note: str = "") -> None:
    row = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        iteration,
        "" if score is None else f"{score:.6f}",
        status,
        retries,
        model,
        temperature,
        hashlib.sha1(code.encode("utf-8")).hexdigest()[:12],
        note.replace("\t", " ").replace("\n", " | ")[:500],
    ]
    with RESULTS_FILE.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(row)


def load_best_score() -> float:
    if not RESULTS_FILE.exists():
        return float("inf")
    best = float("inf")
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score_text = row.get("score", "")
            if not score_text:
                continue
            try:
                score = float(score_text)
            except ValueError:
                continue
            if score < best:
                best = score
    return best


def top_k_results(limit: int = 5) -> str:
    if not RESULTS_FILE.exists():
        return "No previous runs yet."
    rows = []
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            score_text = row.get("score", "")
            if not score_text:
                continue
            try:
                score = float(score_text)
            except ValueError:
                continue
            rows.append((score, row))
    if not rows:
        return "No successful scored runs yet."
    rows.sort(key=lambda x: x[0])
    selected = rows[:limit]
    lines = []
    for score, row in selected:
        lines.append(
            f"iter={row.get('iteration', '?')}, score={score:.5f}, status={row.get('status', '')}, note={row.get('note', '')}"
        )
    return "\n".join(lines)


def ask_ollama(prompt: str, model: str, temperature: float, client: Client | None = None) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        if client is None:
            response = chat(
                model,
                messages=messages,
                options={"temperature": temperature, "num_predict": 8192, "num_ctx": 16384},
            )
        else:
            response = client.chat(
                model,
                messages=messages,
                options={"temperature": temperature, "num_predict": 8192, "num_ctx": 16384},
            )
    except ResponseError as exc:
        if exc.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' not found. Run: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"Ollama API error: {exc.error}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    message = getattr(response, "message", None)
    if message is None:
        raise RuntimeError("Unexpected Ollama response: missing message field")

    thinking_text = (getattr(message, "thinking", None) or "").strip()
    final_text = (getattr(message, "content", None) or "").strip()
    if not final_text:
        if thinking_text:
            print(f"Warning: Empty final content, falling back to thinking_text (len={len(thinking_text)})", file=sys.stderr)
            final_text = thinking_text
        else:
            raise RuntimeError(
                f"Unexpected Ollama response: empty final content (thinking_len={len(thinking_text)})"
            )
    return final_text


def extract_json_hyperparams(raw_response: str) -> str:
    text = raw_response.replace('

', '
')
    blocks = re.findall(r'`(?:json)?
(.*?)(?:`|$)', text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        longest_block = max(blocks, key=len)
        candidate = longest_block.strip()
    else:
        match = re.search(r'\{.*\}', text, flags=re.DOTALL)
        if match:
            candidate = match.group(0).strip()
        else:
            candidate = '{}'
    return candidate

def run_experiment(python_exe: str) -> tuple[int, str, str]:
    result = subprocess.run([python_exe, str(TARGET_FILE)], capture_output=True, text=True, cwd=DATA_DIR)
    return result.returncode, result.stdout, result.stderr


def parse_rmse(stdout: str) -> float | None:
    match = re.search(r"val_rmse:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", stdout)
    if not match:
        return None
    return float(match.group(1))


def generation_prompt(current_code: str, best_history: str) -> str:
    return f"""
You are optimizing a Kaggle training script.

Rules:
1) Return ONLY valid Python source code. Do NOT include <think> tags, explanations, or any text other than the Python code itself. 
2) Begin directly with code. 
3) Keep the script lightweight and focused on improving validation RMSE.
4) The script must print exactly one line: val_rmse: <number>
5) It must load X_train.npy, y_train.npy, X_val.npy, y_val.npy from disk.
6) Prioritize small, targeted model and hyperparameter changes.

Top 5 historical best runs:
{best_history}

Current train.py:
{current_code}
""".strip()


def repair_prompt(stderr: str, broken_code: str) -> str:
    return f"""
Your previous train.py crashed.
Return ONLY a fully corrected train.py as plain Python code.
Do NOT include <think> tags or explanations. 

The script must print exactly one line: val_rmse: <number>

Crash stderr:
{stderr}

Broken train.py:
{broken_code}
""".strip()


def run_loop(iterations: int, temperature: float, model: str, python_exe: str, pause_s: float, ollama_host: str) -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    client = Client(host=ollama_host) if ollama_host else None
    ensure_results_header()
    BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    best_score = load_best_score()
    iteration = 0

    while iterations <= 0 or iteration < iterations:
        iteration += 1
        current_code = read_text(TARGET_FILE)
        shutil.copy(TARGET_FILE, BACKUP_FILE)

        best_history = top_k_results(limit=5)
        prompt = generation_prompt(current_code, best_history)

        print(f"[iter {iteration}] requesting candidate from {model}...")
        try:
            raw = ask_ollama(prompt, model=model, temperature=temperature, client=client)
        except Exception as exc:
            print(f"[iter {iteration}] Ollama request failed: {exc}")
            append_result(iteration, None, "llm_error", 0, model, temperature, current_code, str(exc))
            if pause_s > 0:
                time.sleep(pause_s)
            continue
        candidate = extract_python_code(raw)

        if not candidate.strip():
            print(f"[iter {iteration}] empty candidate, reverting")
            shutil.copy(BACKUP_FILE, TARGET_FILE)
            append_result(iteration, None, "invalid", 0, model, temperature, current_code, "empty model response")
            continue

        write_text(TARGET_FILE, candidate)

        ret_code, stdout, stderr = run_experiment(python_exe)
        retries = 0

        while ret_code != 0 and retries < 3:
            retries += 1
            print(f"[iter {iteration}] crash detected, self-heal attempt {retries}/3")
            try:
                fix_raw = ask_ollama(
                    repair_prompt(stderr, read_text(TARGET_FILE)),
                    model=model,
                    temperature=temperature,
                    client=client,
                )
            except Exception as exc:
                stderr = f"Self-heal request failed: {exc}\nOriginal stderr:\n{stderr}"
                break
            fixed_code = extract_python_code(fix_raw)
            if not fixed_code.strip():
                break
            write_text(TARGET_FILE, fixed_code)
            ret_code, stdout, stderr = run_experiment(python_exe)

        if ret_code != 0:
            print(f"[iter {iteration}] failed after retries, reverting")
            broken = read_text(TARGET_FILE)
            shutil.copy(BACKUP_FILE, TARGET_FILE)
            append_result(iteration, None, "crash", retries, model, temperature, broken, stderr[-500:])
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        score = parse_rmse(stdout)
        if score is None:
            print(f"[iter {iteration}] could not parse val_rmse, reverting")
            candidate_now = read_text(TARGET_FILE)
            shutil.copy(BACKUP_FILE, TARGET_FILE)
            append_result(iteration, None, "parse_fail", retries, model, temperature, candidate_now, stdout[-500:])
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        if score < best_score:
            best_score = score
            print(f"[iter {iteration}] new best: {score:.5f}")
            append_result(iteration, score, "kept", retries, model, temperature, read_text(TARGET_FILE), "")
        else:
            print(f"[iter {iteration}] score {score:.5f} worse than best {best_score:.5f}; reverting")
            candidate_now = read_text(TARGET_FILE)
            shutil.copy(BACKUP_FILE, TARGET_FILE)
            append_result(iteration, score, "reverted", retries, model, temperature, candidate_now, f"best={best_score:.5f}")

        if pause_s > 0:
            time.sleep(pause_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous local Kaggle optimizer using Ollama + train.py")
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--ollama-host", type=str, default="")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between iterations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_loop(
        iterations=args.iterations,
        temperature=args.temperature,
        model=args.model,
        python_exe=args.python_exe,
        pause_s=args.pause,
        ollama_host=args.ollama_host,
    )


if __name__ == "__main__":
    main()
