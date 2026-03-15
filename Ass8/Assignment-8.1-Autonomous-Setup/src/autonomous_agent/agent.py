from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

DEFAULT_TRAIN_TIMEOUT_S = 60.0
DEFAULT_OLLAMA_TIMEOUT_S = 90.0
DEFAULT_MAX_GRID_SIZE = 128
MAX_VALUES_PER_PARAM = 5

HYPERPARAM_SPECS: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 50, "max": 2000, "default": [1000]},
    "max_depth": {"type": "int", "min": 3, "max": 10, "default": [4]},
    "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": [0.05]},
    "subsample": {"type": "float", "min": 0.5, "max": 1.0, "default": [1.0]},
    "colsample_bytree": {"type": "float", "min": 0.5, "max": 1.0, "default": [1.0]},
    "gamma": {"type": "float", "min": 0.0, "max": 5.0, "default": [0.0]},
    "reg_alpha": {"type": "float", "min": 0.0, "max": 5.0, "default": [0.0]},
    "reg_lambda": {"type": "float", "min": 0.0, "max": 5.0, "default": [1.0]},
}

TOGGLE_DEFAULTS: dict[str, bool] = {
    "add_squared_features": False,
    "use_hist_tree_method": False,
}


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


def mixed_history_context(best_limit: int = 3, failure_limit: int = 2) -> str:
    title_best = f"TOP {best_limit} BEST-SCORING RUNS (lowest RMSE):"
    title_fail = f"LAST {failure_limit} FAILED/NO-SCORE RUNS (the wall of shame):"

    if not RESULTS_FILE.exists():
        return "\n".join([title_best, "- none", "", title_fail, "- none"])

    scored_rows: list[tuple[float, int, str, str]] = []
    failed_rows: list[tuple[int, str, str]] = []

    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            status = (row.get("status", "") or "").strip()
            note = (row.get("note", "") or "").strip()

            iteration = -1
            try:
                iteration = int((row.get("iteration", "") or "").strip())
            except ValueError:
                iteration = -1

            score_text = (row.get("score", "") or "").strip()
            if not score_text:
                failed_rows.append((iteration, status, note))
                continue

            try:
                score = float(score_text)
            except ValueError:
                failed_rows.append((iteration, status, note or f"unparseable_score={score_text}"))
                continue

            scored_rows.append((score, iteration, status, note))

    scored_rows.sort(key=lambda item: item[0])
    failed_rows.sort(key=lambda item: item[0], reverse=True)

    lines = [title_best]
    if not scored_rows:
        lines.append("- none")
    else:
        for score, iteration, status, note in scored_rows[:best_limit]:
            lines.append(
                f"- iter={iteration}, score={score:.5f}, status={status or 'unknown'}, reflection={note or '<empty>'}"
            )

    lines.extend(["", title_fail])
    if not failed_rows:
        lines.append("- none")
    else:
        for iteration, status, note in failed_rows[:failure_limit]:
            lines.append(
                f"- iter={iteration}, status={status or 'unknown'}, error_or_note={note or '<empty>'}"
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


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(text)
    except Exception as exc:
        return None, f"JSON parse error: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON root must be object, got {type(value).__name__}"
    return value, "ok"


def validate_json_object(text: str) -> tuple[bool, str]:
    value, note = parse_json_object(text)
    return value is not None, note


def _coerce_param_value(key: str, raw: Any, spec: dict[str, Any]) -> float | int:
    if isinstance(raw, bool):
        raise ValueError(f"{key} cannot use bool values")

    expected = spec["type"]
    if expected == "int":
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{key} must use integer values")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} has non-integer value: {raw!r}") from exc
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} has non-float value: {raw!r}") from exc

    lower = spec["min"]
    upper = spec["max"]
    return min(max(value, lower), upper)


def normalize_switchboard_payload(raw_payload: dict[str, Any], max_grid_size: int = DEFAULT_MAX_GRID_SIZE) -> tuple[dict[str, Any] | None, str]:
    allowed = set(TOGGLE_DEFAULTS) | set(HYPERPARAM_SPECS)
    unknown = sorted(set(raw_payload) - allowed)
    if unknown:
        return None, f"Unknown keys: {', '.join(unknown)}"

    normalized: dict[str, Any] = {}
    for key, default in TOGGLE_DEFAULTS.items():
        raw_value = raw_payload.get(key, default)
        if not isinstance(raw_value, bool):
            return None, f"Toggle '{key}' must be boolean"
        normalized[key] = raw_value

    grid_size = 1
    for key, spec in HYPERPARAM_SPECS.items():
        raw_values = raw_payload.get(key, spec["default"])
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if not values:
            return None, f"{key} list cannot be empty"
        if len(values) > MAX_VALUES_PER_PARAM:
            return None, f"{key} list exceeds max length {MAX_VALUES_PER_PARAM}"

        try:
            normalized_values = sorted({_coerce_param_value(key, value, spec) for value in values})
        except ValueError as exc:
            return None, str(exc)

        if not normalized_values:
            return None, f"{key} list had no usable values"
        normalized[key] = normalized_values
        grid_size *= len(normalized_values)

    if grid_size > max_grid_size:
        return None, f"Grid size {grid_size} exceeds cap {max_grid_size}"
    return normalized, f"ok grid_size={grid_size}"


def canonical_prompt_hyperparams(raw_text: str, max_grid_size: int = DEFAULT_MAX_GRID_SIZE) -> str:
    """
    Format CURRENT HYPERPARAMS JSON in canonical switchboard shape so the model
    sees the same schema we ask it to return (toggles + list-valued params).
    """
    safe_defaults: dict[str, Any] = {**TOGGLE_DEFAULTS}
    for key, spec in HYPERPARAM_SPECS.items():
        default_values = spec.get("default", [])
        if isinstance(default_values, list) and default_values:
            safe_defaults[key] = [default_values[0]]
        else:
            safe_defaults[key] = [spec["min"]]

    parsed, _ = parse_json_object(raw_text)
    if parsed is None:
        return json.dumps(safe_defaults, indent=2, sort_keys=True)

    # Accept both plain switchboard payloads and wrapped {reflection, configuration} objects.
    payload = parsed
    if isinstance(payload.get("configuration"), dict):
        payload = payload["configuration"]

    normalized, _ = normalize_switchboard_payload(payload, max_grid_size=max_grid_size)
    if normalized is None:
        return json.dumps(safe_defaults, indent=2, sort_keys=True)
    return json.dumps(normalized, indent=2, sort_keys=True)


def _call_ollama(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    client: Client | None,
) -> Any:
    if client is None:
        return chat(
            model,
            messages=messages,
            options={"temperature": temperature, "num_predict": 8192, "num_ctx": 16384},
            think=True,
        )
    return client.chat(
        model,
        messages=messages,
        options={"temperature": temperature, "num_predict": 8192, "num_ctx": 16384},
        think=True,
    )


def ask_ollama(
    prompt: str,
    model: str,
    temperature: float,
    client: Client | None = None,
    verbose: bool = False,
    debug_label: str = "",
    debug_log_path: Path | None = None,
    request_timeout_s: float = DEFAULT_OLLAMA_TIMEOUT_S,
) -> str:
    messages = [{"role": "user", "content": prompt}]

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_call_ollama, model, messages, temperature, client)
    try:
        response = future.result(timeout=request_timeout_s)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Ollama request timed out after {request_timeout_s:.1f}s") from exc
    except ResponseError as exc:
        if exc.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' not found. Run: ollama pull {model}"
            ) from exc
        raise RuntimeError(f"Ollama API error: {exc.error}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

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


def parse_reflection_configuration(
    candidate_json: str,
    max_grid_size: int = DEFAULT_MAX_GRID_SIZE,
) -> tuple[dict[str, Any] | None, str | None, str]:
    parsed_candidate, parse_note = parse_json_object(candidate_json)
    if parsed_candidate is None:
        return None, None, parse_note

    required_keys = {"reflection", "configuration"}
    current_keys = set(parsed_candidate.keys())
    if current_keys != required_keys:
        missing = sorted(required_keys - current_keys)
        extra = sorted(current_keys - required_keys)
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"extra={','.join(extra)}")
        suffix = f" ({'; '.join(details)})" if details else ""
        return None, None, f"Candidate must contain exactly root keys 'reflection' and 'configuration'{suffix}"

    reflection_value = parsed_candidate.get("reflection")
    if not isinstance(reflection_value, str):
        return None, None, "reflection must be a string"
    if not reflection_value.strip():
        return None, None, "reflection must be non-empty"

    configuration_value = parsed_candidate.get("configuration")
    if not isinstance(configuration_value, dict):
        return None, reflection_value, "configuration must be a JSON object"

    normalized_candidate, normalize_note = normalize_switchboard_payload(configuration_value, max_grid_size=max_grid_size)
    if normalized_candidate is None:
        return None, reflection_value, f"configuration validation failed: {normalize_note}"

    return normalized_candidate, reflection_value, "ok"


def with_reflection_detail(reflection: str, detail: str) -> str:
    reflection_text = reflection.strip()
    detail_text = detail.strip()
    if reflection_text and detail_text:
        return f"{reflection_text} | {detail_text}"
    if reflection_text:
        return reflection_text
    return detail_text


def _decoded_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_experiment(python_exe: str, timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S) -> tuple[int, str, str, bool]:
    try:
        result = subprocess.run(
            [python_exe, str(TARGET_FILE)],
            capture_output=True,
            text=True,
            cwd=DATA_DIR,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decoded_text(exc.stdout)
        stderr = _decoded_text(exc.stderr)
        timeout_note = f"Training timed out after {timeout_s:.1f}s"
        joined_stderr = "\n".join(part for part in (stderr.strip(), timeout_note) if part).strip()
        return 124, stdout, joined_stderr, True
    return result.returncode, result.stdout, result.stderr, False


def _normalize_experiment_result(result: Any) -> tuple[int, str, str, bool]:
    if isinstance(result, tuple):
        if len(result) == 4:
            ret_code, stdout, stderr, timed_out = result
            return int(ret_code), str(stdout), str(stderr), bool(timed_out)
        if len(result) == 3:
            ret_code, stdout, stderr = result
            return int(ret_code), str(stdout), str(stderr), False
    raise TypeError(f"Unexpected run_experiment result: {result!r}")


def parse_rmse(stdout: str) -> float | None:
    match = re.search(r"val_rmse:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", stdout)
    if not match:
        return None
    return float(match.group(1))


def parse_training_result(stdout: str) -> tuple[float | None, str, dict[str, Any] | None]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        parsed_obj, _ = parse_json_object(line)
        if parsed_obj is None:
            continue
        status = str(parsed_obj.get("status", "")).strip()
        if not status:
            continue
        score_raw = parsed_obj.get("score", parsed_obj.get("val_rmse"))
        score: float | None = None
        if isinstance(score_raw, (int, float)) and not isinstance(score_raw, bool):
            score = float(score_raw)
        return score, status, parsed_obj

    legacy_score = parse_rmse(stdout)
    if legacy_score is not None:
        return legacy_score, "ok_legacy", {"status": "ok_legacy", "score": legacy_score}
    return None, "parse_fail", None


def summarize_training_payload(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return ""
    status = payload.get("status", "")
    grid_size = payload.get("grid_size")
    elapsed_seconds = payload.get("elapsed_seconds")
    parts = [f"train_status={status}"]
    if isinstance(grid_size, int):
        parts.append(f"grid_size={grid_size}")
    if isinstance(elapsed_seconds, (int, float)):
        parts.append(f"elapsed_s={float(elapsed_seconds):.3f}")
    best_params = payload.get("best_params")
    if isinstance(best_params, dict) and best_params:
        compact = json.dumps(best_params, sort_keys=True)
        parts.append(f"best_params={compact}")
    return "; ".join(parts)


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


def generation_prompt(current_code: str, history_context: str, max_grid_size: int = DEFAULT_MAX_GRID_SIZE) -> str:
    no_history = "score=" not in history_context
    if no_history:
        intro = (
            "You are an AI optimizing XGBoost hyperparameters for a Kaggle dataset. "
            "Generate one candidate JSON based on defaults and avoid known failure patterns."
        )
    else:
        intro = (
            "You are an AI optimizing XGBoost hyperparameters for a Kaggle dataset. "
            "Read your historical reflections and failures, then choose parameters to minimize validation RMSE."
        )

    return f'''
{intro}

You are a pilot-only model. Do not write Python code or explanations.
Return exactly one JSON object with exactly two root keys:
- reflection: 1-2 sentences explaining what you learned from prior runs and why this proposal should help.
- configuration: object containing ONLY the keys below.

The configuration object must contain:

Boolean toggles:
- add_squared_features: bool
- use_hist_tree_method: bool

Hyperparameter lists (must be JSON lists with exactly 1 value per key for this run):
- n_estimators: int list (50 to 2000)
- max_depth: int list (3 to 10)
- learning_rate: float list (0.01 to 0.3)
- subsample: float list (0.5 to 1.0)
- colsample_bytree: float list (0.5 to 1.0)
- gamma: float list (0 to 5)
- reg_alpha: float list (0 to 5)
- reg_lambda: float list (0 to 5)

For this run, always use exactly one value in each hyperparameter list.
Do not generate multi-value lists.
This guarantees grid_size=1 and automatically satisfies the cap <= {max_grid_size}.

Output ONLY a valid JSON object wrapped in ```json ... ```.
Do not include any extra root keys and do not omit either required root key.

HISTORY CONTEXT:
{history_context}

CURRENT HYPERPARAMS JSON (canonical switchboard shape):
{current_code}
'''.strip()


def repair_prompt(stderr: str, broken_code: str, history_context: str, max_grid_size: int = DEFAULT_MAX_GRID_SIZE) -> str:
    return f'''
Your previous JSON candidate failed validation or training.
Return ONLY valid JSON wrapped in ```json ... ```.

The JSON must have exactly two root keys:
- reflection: 1-2 sentences on what failed and why this fix should work.
- configuration: object with the allowed switchboard keys and value rules.

Rules:
- Allowed keys only: add_squared_features, use_hist_tree_method,
  n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
- Toggle values must be booleans.
- Hyperparameter values must be JSON lists with exactly 1 value per key.
- Do not generate multi-value lists (this keeps grid_size=1 and <= {max_grid_size}).
- No extra root keys are allowed.

HISTORY CONTEXT:
{history_context}

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
    train_timeout_s: float = DEFAULT_TRAIN_TIMEOUT_S,
    ollama_timeout_s: float = DEFAULT_OLLAMA_TIMEOUT_S,
    max_grid_size: int = DEFAULT_MAX_GRID_SIZE,
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
                f"train_timeout_s={train_timeout_s}, ollama_timeout_s={ollama_timeout_s}, "
                f"max_grid_size={max_grid_size}, "
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
        prompt_current_code = canonical_prompt_hyperparams(current_code, max_grid_size=max_grid_size)
        reflection_text = ""

        history_context = mixed_history_context(best_limit=3, failure_limit=2)
        prompt = generation_prompt(prompt_current_code, history_context, max_grid_size=max_grid_size)
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
                request_timeout_s=ollama_timeout_s,
            )
        except TimeoutError as exc:
            print(f"[iter {iteration}] Ollama request timed out: {exc}")
            debug_log(verbose, f"Ollama request timeout: {exc}", iteration, debug_log_path)
            append_result(iteration, None, "llm_timeout", 0, model, temperature, current_code, str(exc))
            if pause_s > 0:
                time.sleep(pause_s)
            continue
        except Exception as exc:
            print(f"[iter {iteration}] Ollama request failed: {exc}")
            debug_log(verbose, f"Ollama request exception: {exc}", iteration, debug_log_path)
            append_result(iteration, None, "llm_error", 0, model, temperature, current_code, str(exc))
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        debug_block(verbose, "raw_generation_response", raw, iteration, debug_log_path)
        extracted = extract_json_hyperparams(raw)
        debug_block(verbose, "extracted_generation_json", extracted, iteration, debug_log_path)

        normalized_candidate, reflection_candidate, parse_note = parse_reflection_configuration(
            extracted,
            max_grid_size=max_grid_size,
        )
        debug_log(verbose, f"candidate_json_parse={parse_note}", iteration, debug_log_path)
        if normalized_candidate is None:
            print(f"[iter {iteration}] invalid JSON candidate, skipping")
            append_result(
                iteration,
                None,
                "invalid",
                0,
                model,
                temperature,
                current_code,
                with_reflection_detail(reflection_candidate or "", parse_note),
            )
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        reflection_text = reflection_candidate or ""

        candidate = json.dumps(normalized_candidate, indent=2, sort_keys=True)
        write_text(DATA_DIR / "hyperparams.json", candidate)

        raw_result = run_experiment(python_exe, timeout_s=train_timeout_s)
        ret_code, stdout, stderr, timed_out = _normalize_experiment_result(raw_result)
        retries = 0
        debug_log(verbose, f"experiment_attempt=0 return_code={ret_code} timed_out={timed_out}", iteration, debug_log_path)
        debug_block(verbose, "experiment_stdout_attempt_0", stdout, iteration, debug_log_path)
        debug_block(verbose, "experiment_stderr_attempt_0", stderr, iteration, debug_log_path)

        if timed_out:
            timeout_note = stderr[-500:] if stderr.strip() else f"timeout_after={train_timeout_s:.1f}s"
            print(f"[iter {iteration}] training timed out, skipping candidate")
            append_result(
                iteration,
                None,
                "timeout",
                retries,
                model,
                temperature,
                candidate,
                with_reflection_detail(reflection_text, timeout_note),
            )
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        if ret_code != 0:
            fatal_error_type = detect_fatal_execution_error(stderr)
            if fatal_error_type is not None:
                fatal_message = (
                    f"[iter {iteration}] fatal execution error detected ({fatal_error_type}); "
                    "stopping run for manual fix."
                )
                print(fatal_message)
                debug_log(verbose, fatal_message, iteration, debug_log_path)
                append_result(
                    iteration,
                    None,
                    "crash",
                    retries,
                    model,
                    temperature,
                    candidate,
                    with_reflection_detail(reflection_text, f"fatal_error={fatal_error_type}; stderr_tail={stderr[-500:]}"),
                )
                raise RuntimeError(fatal_message)

        while ret_code != 0 and retries < 3:
            retries += 1
            print(f"[iter {iteration}] crash detected, self-heal attempt {retries}/3")
            broken_code = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            repair_msg = repair_prompt(stderr, broken_code, history_context, max_grid_size=max_grid_size)
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
                    request_timeout_s=ollama_timeout_s,
                )
            except TimeoutError as exc:
                stderr = f"Self-heal request timed out: {exc}\nOriginal stderr:\n{stderr}"
                debug_log(verbose, f"Self-heal request timeout: {exc}", iteration, debug_log_path)
                break
            except Exception as exc:
                stderr = f"Self-heal request failed: {exc}\nOriginal stderr:\n{stderr}"
                debug_log(verbose, f"Self-heal request exception: {exc}", iteration, debug_log_path)
                break

            debug_block(verbose, f"repair_raw_response_attempt_{retries}", fix_raw, iteration, debug_log_path)
            fix_extracted = extract_json_hyperparams(fix_raw)
            debug_block(verbose, f"repair_extracted_json_attempt_{retries}", fix_extracted, iteration, debug_log_path)

            normalized_fixed, fixed_reflection, parsed_fixed_note = parse_reflection_configuration(
                fix_extracted,
                max_grid_size=max_grid_size,
            )
            debug_log(verbose, f"repair_json_parse attempt={retries}: {parsed_fixed_note}", iteration, debug_log_path)
            if normalized_fixed is None:
                stderr = f"Repair candidate is invalid JSON: {parsed_fixed_note}"
                break

            reflection_text = fixed_reflection or reflection_text

            fixed_code = json.dumps(normalized_fixed, indent=2, sort_keys=True)
            write_text(DATA_DIR / "hyperparams.json", fixed_code)
            raw_result = run_experiment(python_exe, timeout_s=train_timeout_s)
            ret_code, stdout, stderr, timed_out = _normalize_experiment_result(raw_result)
            debug_log(verbose, f"experiment_attempt={retries} return_code={ret_code} timed_out={timed_out}", iteration, debug_log_path)
            debug_block(verbose, f"experiment_stdout_attempt_{retries}", stdout, iteration, debug_log_path)
            debug_block(verbose, f"experiment_stderr_attempt_{retries}", stderr, iteration, debug_log_path)

            if timed_out:
                break

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
                        with_reflection_detail(reflection_text, f"fatal_error={fatal_error_type}; stderr_tail={stderr[-500:]}"),
                    )
                    raise RuntimeError(fatal_message)

        if timed_out:
            timeout_note = stderr[-500:] if stderr.strip() else f"timeout_after={train_timeout_s:.1f}s"
            print(f"[iter {iteration}] training timed out, skipping candidate")
            candidate_now = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            append_result(
                iteration,
                None,
                "timeout",
                retries,
                model,
                temperature,
                candidate_now,
                with_reflection_detail(reflection_text, timeout_note),
            )
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        if ret_code != 0:
            print(f"[iter {iteration}] failed after retries, reverting")
            broken = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            _, train_status, train_payload = parse_training_result(stdout)
            payload_note = summarize_training_payload(train_payload)
            merged_note = "; ".join(part for part in [payload_note, stderr[-500:]] if part)
            crash_status = train_status if train_status not in {"ok", "ok_legacy", "parse_fail"} else "crash"
            debug_log(verbose, f"Iteration ended in failure after retry budget. status={crash_status}", iteration, debug_log_path)
            append_result(
                iteration,
                None,
                crash_status,
                retries,
                model,
                temperature,
                broken,
                with_reflection_detail(reflection_text, merged_note),
            )
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        score, train_status, train_payload = parse_training_result(stdout)
        payload_note = summarize_training_payload(train_payload)
        if score is None or train_status not in {"ok", "ok_legacy"}:
            print(f"[iter {iteration}] could not parse successful training score, status={train_status}")
            candidate_now = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"
            debug_log(verbose, f"train_result_parse_status={train_status}; payload_note={payload_note}", iteration, debug_log_path)
            status = train_status if train_status != "ok" else "parse_fail"
            append_result(
                iteration,
                None,
                status,
                retries,
                model,
                temperature,
                candidate_now,
                with_reflection_detail(reflection_text, payload_note or stdout[-500:]),
            )
            if pause_s > 0:
                time.sleep(pause_s)
            continue

        debug_log(verbose, f"Parsed score={score:.6f}; train_status={train_status}", iteration, debug_log_path)
        candidate_now = read_text(DATA_DIR / "hyperparams.json") if (DATA_DIR / "hyperparams.json").exists() else "{}"

        if score < best_score:
            best_score = score
            print(f"[iter {iteration}] new best: {score:.5f}")
            debug_log(verbose, f"Iteration result=kept, updated_best={best_score:.6f}", iteration, debug_log_path)
            append_result(iteration, score, "kept", retries, model, temperature, candidate_now, reflection_text)
        else:
            print(f"[iter {iteration}] score {score:.5f} worse than best {best_score:.5f}; reverting")
            debug_log(
                verbose,
                f"Iteration result=reverted, score={score:.6f}, best={best_score:.6f}",
                iteration,
                debug_log_path,
            )
            append_result(iteration, score, "reverted", retries, model, temperature, candidate_now, reflection_text)

        if pause_s > 0:
            time.sleep(pause_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous local Kaggle optimizer using Ollama + JSON switchboard")
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--ollama-host", type=str, default="")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between iterations")
    parser.add_argument("--train-timeout", type=float, default=DEFAULT_TRAIN_TIMEOUT_S, help="Timeout in seconds for each train.py run")
    parser.add_argument("--ollama-timeout", type=float, default=DEFAULT_OLLAMA_TIMEOUT_S, help="Timeout in seconds for each Ollama request")
    parser.add_argument("--max-grid-size", type=int, default=DEFAULT_MAX_GRID_SIZE, help="Maximum allowed parameter combinations per iteration")
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
        train_timeout_s=args.train_timeout,
        ollama_timeout_s=args.ollama_timeout,
        max_grid_size=args.max_grid_size,
    )


if __name__ == "__main__":
    main()
