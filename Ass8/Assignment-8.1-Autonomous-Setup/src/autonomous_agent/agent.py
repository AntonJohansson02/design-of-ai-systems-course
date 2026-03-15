from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
REQUIRED_TRAIN_PACKAGES = ("numpy", "xgboost", "sklearn")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def ensure_results_header() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        return
    header = ["timestamp_utc", "iteration", "score", "status", "retries", "model", "temperature", "param_hash", "note"]
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


def append_debug_log(log_path: Path | None, text: str) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def debug_log(enabled: bool, message: str, iteration: int | None = None, log_path: Path | None = None) -> None:
    if not enabled:
        return
    prefix = "[debug] " if iteration is None else f"[iter {iteration}] [debug] "
    line = f"{prefix}{message}"
    print(line)
    append_debug_log(log_path, line)


def debug_block(enabled: bool, title: str, content: str, iteration: int | None = None, log_path: Path | None = None) -> None:
    if not enabled:
        return
    prefix = "[debug] " if iteration is None else f"[iter {iteration}] [debug] "
    body = content.strip("\n")
    payload = body if body else "<empty>"
    block = f"{prefix}{title} BEGIN\n{payload}\n{prefix}{title} END"
    print(block)
    append_debug_log(log_path, block)


def validate_json_object(text: str) -> tuple[bool, str]:
    try:
        value = json.loads(text)
    except Exception as exc:
        return False, f"JSON parse error: {exc}"
    if not isinstance(value, dict):
        return False, f"JSON root must be object, got {type(value).__name__}"
    return True, "ok"


def ask_ollama(
    prompt: str,
    model: str,
    temperature: float,
    client: Client | None = None,
    verbose: bool = False,
    debug_label: str = "",
    debug_log_path: Path | None = None,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        if client is None:
            response = chat(
                model,
                messages=messages,
                options={"temperature": temperature, "num_predict": 8192, "num_ctx": 16384},
                think=True,
            )
        else:
            response = client.chat(
                model,
                messages=messages,
                options={"temperature": temperature, "num_predict": 8192, "num_ctx": 16384},
                think=True,
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

    if verbose:
        label = f"{debug_label} " if debug_label else ""
        debug_log(
            True,
            f"{label}ollama message lengths: content_len={len(final_text)}, thinking_len={len(thinking_text)}",
            log_path=debug_log_path,
        )
        debug_block(True, f"{label}ollama.message.content", final_text, log_path=debug_log_path)
        debug_block(True, f"{label}ollama.message.thinking", thinking_text, log_path=debug_log_path)

    if not final_text:
        raise RuntimeError(
            f"Unexpected Ollama response: empty final content (thinking_len={len(thinking_text)})"
        )
    return final_text


def extract_json_hyperparams(raw_response: str) -> str:
    text = raw_response.replace("\r\n", "\n")
    blocks = re.findall(r"```(?:json)?\n(.*?)(?:```|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        longest_block = max(blocks, key=len)
        candidate = longest_block.strip()
    else:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0).strip()
        else:
            candidate = "{}"
    return candidate

def run_experiment(python_exe: str) -> tuple[int, str, str]:
    result = subprocess.run([python_exe, str(TARGET_FILE)], capture_output=True, text=True, cwd=DATA_DIR)
    return result.returncode, result.stdout, result.stderr


def parse_rmse(stdout: str) -> float | None:
    match = re.search(r"val_rmse:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", stdout)
    if not match:
        return None
    return float(match.group(1))


def detect_fatal_execution_error(stderr: str) -> str | None:
    fatal_error_types = ("ModuleNotFoundError", "ImportError", "SyntaxError", "IndentationError")
    for error_type in fatal_error_types:
        if re.search(rf"\b{re.escape(error_type)}\b", stderr):
            return error_type
    return None


def verify_training_environment(python_exe: str) -> tuple[bool, str]:
    check_script = (
        "import importlib.util, sys; "
        "pkgs=('numpy','xgboost','sklearn'); "
        "missing=[pkg for pkg in pkgs if importlib.util.find_spec(pkg) is None]; "
        "print(','.join(missing)); "
        "raise SystemExit(0 if not missing else 3)"
    )

    try:
        result = subprocess.run([python_exe, "-c", check_script], capture_output=True, text=True)
    except Exception as exc:
        return False, f"Failed to run environment check with '{python_exe}': {exc}"

    if result.returncode == 0:
        return True, "ok"

    missing = [item.strip() for item in result.stdout.strip().split(",") if item.strip()]
    if missing:
        return (
            False,
            (
                "Missing required training dependencies in the selected Python environment: "
                f"{', '.join(missing)}. Install them in '{python_exe}' before running the agent."
            ),
        )

    detail = (result.stderr or result.stdout).strip()
    return False, f"Environment check failed for '{python_exe}'. Details: {detail}"


def generation_prompt(current_code: str, best_history: str) -> str:
    no_history = best_history in {"No previous runs yet.", "No successful scored runs yet."}
    if no_history:
        intro = (
            "You are an AI optimizing XGBoost hyperparameters for a Kaggle dataset. "
            "Generate one candidate JSON within allowed hyperparameter bounds."
        )
    else:
        intro = (
            "You are an AI optimizing XGBoost hyperparameters for a Kaggle dataset. "
            "Examine past experiments and choose new parameters to minimize validation RMSE."
        )

    return f'''
{intro}

Current allowed hyperparameters:
- n_estimators: int (50 to 2000)
- max_depth: int (3 to 10)
- learning_rate: float (0.01 to 0.3)
- subsample: float (0.5 to 1.0)
- colsample_bytree: float (0.5 to 1.0)
- gamma: float (0 to 5)
- reg_alpha: float (0 to 5)
- reg_lambda: float (0 to 5)

Output ONLY a valid JSON object wrapped in ```json ... ```. 

TOP 5 HISTORICAL BEST RUNS:
{best_history}

CURRENT HYPERPARAMS JSON:
{current_code}
'''.strip()


def repair_prompt(stderr: str, broken_code: str) -> str:
    return f'''
Your previous JSON hyperparameters caused a crash or failed to parse.
Return ONLY valid JSON wrapped in ```json ... ```.

CRASH STDERR:
{stderr}

BROKEN JSON:
{broken_code}
'''.strip()


def run_loop(
    iterations: int,
    temperature: float,
    model: str,
    python_exe: str,
    pause_s: float,
    ollama_host: str,
    verbose: bool = False,
    verbose_log_file: str = "",
) -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing target file: {TARGET_FILE}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")

    debug_log_path: Path | None = None
    if verbose_log_file.strip():
        debug_log_path = Path(verbose_log_file.strip())
        if not debug_log_path.is_absolute():
            debug_log_path = PROJECT_ROOT / debug_log_path
    if verbose:
        debug_log(
            True,
            (
                f"Verbose logging enabled. model={model}, temperature={temperature}, "
                f"python_exe={python_exe}, ollama_host={ollama_host or '<default>'}, "
                f"log_file={debug_log_path or '<stdout-only>'}"
            ),
            log_path=debug_log_path,
        )

    env_ok, env_note = verify_training_environment(python_exe)
    if not env_ok:
        debug_log(verbose, f"training_env_check=failed detail={env_note}", log_path=debug_log_path)
        raise RuntimeError(env_note)
    debug_log(verbose, f"training_env_check=ok packages={', '.join(REQUIRED_TRAIN_PACKAGES)}", log_path=debug_log_path)

    client = Client(host=ollama_host) if ollama_host else None
    ensure_results_header()
    BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    best_score = load_best_score()
    iteration = 0

    while iterations <= 0 or iteration < iterations:
        iteration += 1
        current_code = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"

        best_history = top_k_results(limit=5)
        prompt = generation_prompt(current_code, best_history)
        debug_block(verbose, "generation_prompt", prompt, iteration, debug_log_path)

        print(f"[iter {iteration}] requesting candidate from {model}...")
        try:
            raw = ask_ollama(
                prompt,
                model=model,
                temperature=temperature,
                client=client,
                verbose=verbose,
                debug_label=f"iter {iteration} generation",
                debug_log_path=debug_log_path,
            )
        except Exception as exc:
            print(f"[iter {iteration}] Ollama request failed: {exc}")
            debug_log(verbose, f"Ollama request exception: {exc}", iteration, debug_log_path)
            append_result(iteration, None, "llm_error", 0, model, temperature, current_code, str(exc))
            if pause_s > 0:
                time.sleep(pause_s)
            continue
        debug_block(verbose, "raw_generation_response", raw, iteration, debug_log_path)
        candidate = extract_json_hyperparams(raw)
        debug_block(verbose, "extracted_generation_json", candidate, iteration, debug_log_path)
        valid_candidate, valid_candidate_note = validate_json_object(candidate)
        debug_log(
            verbose,
            f"candidate_json_valid={valid_candidate}; detail={valid_candidate_note}",
            iteration,
            debug_log_path,
        )

        if not candidate.strip():
            print(f"[iter {iteration}] empty candidate, reverting")
            debug_log(verbose, "Candidate extraction returned empty text.", iteration, debug_log_path)
            append_result(iteration, None, "invalid", 0, model, temperature, current_code, "empty model response")
            continue

        write_text(DATA_DIR / "hyperparams.json", candidate)

        ret_code, stdout, stderr = run_experiment(python_exe)
        retries = 0
        debug_log(verbose, f"experiment_attempt=0 return_code={ret_code}", iteration, debug_log_path)
        debug_block(verbose, "experiment_stdout_attempt_0", stdout, iteration, debug_log_path)
        debug_block(verbose, "experiment_stderr_attempt_0", stderr, iteration, debug_log_path)

        if ret_code != 0:
            fatal_error_type = detect_fatal_execution_error(stderr)
            if fatal_error_type is not None:
                fatal_message = (
                    f"[iter {iteration}] fatal execution error detected ({fatal_error_type}); "
                    "stopping run for manual fix."
                )
                print(fatal_message)
                debug_log(verbose, fatal_message, iteration, debug_log_path)
                broken = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
                append_result(
                    iteration,
                    None,
                    "crash",
                    retries,
                    model,
                    temperature,
                    broken,
                    f"fatal_error={fatal_error_type}; stderr_tail={stderr[-500:]}",
                )
                raise RuntimeError(fatal_message)

        while ret_code != 0 and retries < 3:
            retries += 1
            print(f"[iter {iteration}] crash detected, self-heal attempt {retries}/3")
            broken_code = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            repair_msg = repair_prompt(stderr, broken_code)
            debug_block(verbose, f"repair_prompt_attempt_{retries}", repair_msg, iteration, debug_log_path)
            try:
                fix_raw = ask_ollama(
                    repair_msg,
                    model=model,
                    temperature=temperature,
                    client=client,
                    verbose=verbose,
                    debug_label=f"iter {iteration} repair attempt {retries}",
                    debug_log_path=debug_log_path,
                )
            except Exception as exc:
                stderr = f"Self-heal request failed: {exc}\nOriginal stderr:\n{stderr}"
                debug_log(verbose, f"Self-heal request exception: {exc}", iteration, debug_log_path)
                break
            debug_block(verbose, f"repair_raw_response_attempt_{retries}", fix_raw, iteration, debug_log_path)
            fixed_code = extract_json_hyperparams(fix_raw)
            debug_block(verbose, f"repair_extracted_json_attempt_{retries}", fixed_code, iteration, debug_log_path)
            valid_fixed, valid_fixed_note = validate_json_object(fixed_code)
            debug_log(
                verbose,
                f"repair_json_valid attempt={retries}: {valid_fixed}; detail={valid_fixed_note}",
                iteration,
                debug_log_path,
            )
            if not fixed_code.strip():
                debug_log(verbose, "Self-heal produced empty JSON text; aborting retries.", iteration, debug_log_path)
                break
            write_text(DATA_DIR / "hyperparams.json", fixed_code)
            ret_code, stdout, stderr = run_experiment(python_exe)
            debug_log(verbose, f"experiment_attempt={retries} return_code={ret_code}", iteration, debug_log_path)
            debug_block(verbose, f"experiment_stdout_attempt_{retries}", stdout, iteration, debug_log_path)
            debug_block(verbose, f"experiment_stderr_attempt_{retries}", stderr, iteration, debug_log_path)
            if ret_code != 0:
                fatal_error_type = detect_fatal_execution_error(stderr)
                if fatal_error_type is not None:
                    fatal_message = (
                        f"[iter {iteration}] fatal execution error detected ({fatal_error_type}); "
                        "stopping run for manual fix."
                    )
                    print(fatal_message)
                    debug_log(verbose, fatal_message, iteration, debug_log_path)
                    broken = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
                    append_result(
                        iteration,
                        None,
                        "crash",
                        retries,
                        model,
                        temperature,
                        broken,
                        f"fatal_error={fatal_error_type}; stderr_tail={stderr[-500:]}",
                    )
                    raise RuntimeError(fatal_message)

        if ret_code != 0:
            print(f"[iter {iteration}] failed after retries, reverting")
            broken = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            debug_log(verbose, "Iteration ended in crash after retry budget.", iteration, debug_log_path)
            append_result(iteration, None, "crash", retries, model, temperature, broken, stderr[-500:])
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        score = parse_rmse(stdout)
        if score is None:
            print(f"[iter {iteration}] could not parse val_rmse, reverting")
            candidate_now = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            debug_log(verbose, "val_rmse parsing failed for successful subprocess run.", iteration, debug_log_path)
            append_result(iteration, None, "parse_fail", retries, model, temperature, candidate_now, stdout[-500:])
            if pause_s > 0:
                time.sleep(pause_s)
            continue
        debug_log(verbose, f"Parsed val_rmse={score:.6f}", iteration, debug_log_path)

        if score < best_score:
            best_score = score
            print(f"[iter {iteration}] new best: {score:.5f}")
            debug_log(verbose, f"Iteration result=kept, updated_best={best_score:.6f}", iteration, debug_log_path)
            append_result(iteration, score, "kept", retries, model, temperature, read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}", "")
        else:
            print(f"[iter {iteration}] score {score:.5f} worse than best {best_score:.5f}; reverting")
            candidate_now = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            debug_log(
                verbose,
                f"Iteration result=reverted, score={score:.6f}, best={best_score:.6f}",
                iteration,
                debug_log_path,
            )
            append_result(iteration, score, "reverted", retries, model, temperature, candidate_now, f"best={best_score:.5f}")

        if pause_s > 0:
            time.sleep(pause_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous local Kaggle optimizer using Ollama + JSON")
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--ollama-host", type=str, default="")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between iterations")
    parser.add_argument(
        "--verbose",
        "--debug",
        action="store_true",
        dest="verbose",
        help="Enable detailed debug output (prompts, raw model output, extracted JSON, and subprocess stdout/stderr).",
    )
    parser.add_argument(
        "--verbose-log-file",
        type=str,
        default="",
        help="Optional log file path for verbose output. Relative paths are resolved from project root.",
    )
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
        verbose=args.verbose,
        verbose_log_file=args.verbose_log_file,
    )


if __name__ == "__main__":
    main()
