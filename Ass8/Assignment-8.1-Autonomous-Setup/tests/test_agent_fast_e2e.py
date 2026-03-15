from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

OLLAMA_HOST = "http://localhost:11434"
FAST_TEST_MODEL = "qwen3.5:2b"


def build_dual_output_candidate(reflection: str, config_overrides: dict | None = None) -> str:
    configuration = {
        "add_squared_features": False,
        "use_hist_tree_method": False,
        "n_estimators": [200],
        "max_depth": [6],
        "learning_rate": [0.07],
        "subsample": [0.9],
        "colsample_bytree": [0.85],
        "gamma": [0.1],
        "reg_alpha": [0.0],
        "reg_lambda": [1.0],
    }
    if config_overrides:
        configuration.update(config_overrides)

    payload = {"reflection": reflection, "configuration": configuration}
    return "```json\n" + json.dumps(payload) + "\n```"


class DummyClient:
    def __init__(self, host: str):
        self.host = host


class DummyResponseError(Exception):
    def __init__(self, error: str = "", status_code: int | None = None):
        super().__init__(error)
        self.error = error
        self.status_code = status_code


def dummy_chat(*args, **kwargs):
    raise RuntimeError("dummy chat should not be called in this patched test path")


def load_agent_module():
    module_path = Path(__file__).resolve().parent.parent / "src" / "autonomous_agent" / "agent.py"
    spec = importlib.util.spec_from_file_location("agent_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    # The test patches all network calls, so a tiny stub keeps import fast and isolated.
    injected_stub = False
    if "ollama" not in sys.modules:
        fake_ollama = types.ModuleType("ollama")
        fake_ollama.Client = DummyClient
        fake_ollama.ResponseError = DummyResponseError
        fake_ollama.chat = dummy_chat
        sys.modules["ollama"] = fake_ollama
        injected_stub = True

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if injected_stub:
            sys.modules.pop("ollama", None)
    return module


class AgentFastE2ETest(unittest.TestCase):
    def test_normalize_switchboard_payload_rejects_unknown_key(self):
        agent = load_agent_module()

        payload, note = agent.normalize_switchboard_payload(
            {
                "n_estimators": [200],
                "max_depth": [6],
                "learning_rate": [0.1],
                "subsample": [0.9],
                "colsample_bytree": [0.9],
                "gamma": [0.0],
                "reg_alpha": [0.0],
                "reg_lambda": [1.0],
                "bad_key": True,
            }
        )

        self.assertIsNone(payload)
        self.assertIn("Unknown keys", note)

    def test_parse_training_result_prefers_structured_payload(self):
        agent = load_agent_module()

        stdout = (
            "debug line\n"
            '{"status":"ok","score":1.2345,"grid_size":4,"best_params":{"max_depth":6}}\n'
        )

        score, status, payload = agent.parse_training_result(stdout)

        self.assertAlmostEqual(score, 1.2345)
        self.assertEqual(status, "ok")
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["grid_size"], 4)

    def test_ollama_client_smoke_qwen2b(self):
        from ollama import chat

        response = chat(
            FAST_TEST_MODEL,
            messages=[{"role": "user", "content": "What is 10 + 23?"}],
            options={"temperature": 0.0},
            think=True,
        )

        thinking_text = (getattr(response.message, "thinking", None) or "").strip()
        final_answer = (response.message.content or "").strip()

        # Validate that reasoning and final answer are both present and separated.
        self.assertTrue(thinking_text)
        self.assertTrue(final_answer)
        self.assertIn("33", final_answer)
        self.assertNotIn("Thinking Process", final_answer)
        self.assertNotEqual(thinking_text, final_answer)

    def test_ask_ollama_uses_final_content_from_same_thinking_call(self):
        agent = load_agent_module()

        fake_response = types.SimpleNamespace(
            message=types.SimpleNamespace(
                thinking="The answer is straightforward arithmetic.",
                content="33",
            )
        )

        with patch.object(agent, "chat", autospec=True, return_value=fake_response) as mock_chat:
            final_answer = agent.ask_ollama(
                "What is 10 + 23?",
                model=FAST_TEST_MODEL,
                temperature=0.0,
                client=None,
            )

        self.assertEqual(final_answer, "33")
        self.assertNotIn("straightforward arithmetic", final_answer)
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        self.assertEqual(args[0], FAST_TEST_MODEL)
        self.assertTrue(kwargs["think"])
        self.assertEqual(kwargs["messages"][0]["content"], "What is 10 + 23?")

    def test_ask_ollama_rejects_empty_final_content(self):
        agent = load_agent_module()

        fake_response = types.SimpleNamespace(
            message=types.SimpleNamespace(
                thinking="I have reasoning but no final channel.",
                content="",
            )
        )

        with patch.object(agent, "chat", autospec=True, return_value=fake_response):
            with self.assertRaises(RuntimeError) as raised:
                agent.ask_ollama(
                    "What is 10 + 23?",
                    model=FAST_TEST_MODEL,
                    temperature=0.0,
                    client=None,
                )

        self.assertIn("empty final content", str(raised.exception))

    def test_generation_prompt_no_history_requires_dual_output_json_schema(self):
        agent = load_agent_module()

        no_history_context = (
            "TOP 3 BEST-SCORING RUNS (lowest RMSE):\n"
            "- none\n\n"
            "LAST 2 FAILED/NO-SCORE RUNS (the wall of shame):\n"
            "- none"
        )
        prompt = agent.generation_prompt("{}", no_history_context)

        self.assertIn("Generate one candidate JSON based on defaults", prompt)
        self.assertIn("exactly two root keys", prompt)
        self.assertIn("- reflection:", prompt)
        self.assertIn("- configuration:", prompt)
        self.assertIn("Output ONLY a valid JSON object wrapped in ```json ... ```.", prompt)

    def test_generation_prompt_with_history_mentions_reflection_and_failure_context(self):
        agent = load_agent_module()

        history_line = "- iter=7, score=0.92345, status=kept, reflection=stable candidate"
        history_context = (
            "TOP 3 BEST-SCORING RUNS (lowest RMSE):\n"
            f"{history_line}\n\n"
            "LAST 2 FAILED/NO-SCORE RUNS (the wall of shame):\n"
            "- iter=8, status=crash, error_or_note=invalid config shape"
        )
        prompt = agent.generation_prompt("{}", history_context)

        self.assertIn("Read your historical reflections and failures", prompt)
        self.assertIn(history_line, prompt)
        self.assertIn("HISTORY CONTEXT", prompt)

    def test_parse_reflection_configuration_enforces_two_root_keys(self):
        agent = load_agent_module()

        candidate = build_dual_output_candidate("I will tighten depth and keep subsample stable.")
        extracted = agent.extract_json_hyperparams(candidate)

        normalized, reflection, note = agent.parse_reflection_configuration(extracted)

        self.assertEqual(note, "ok")
        self.assertEqual(reflection, "I will tighten depth and keep subsample stable.")
        self.assertEqual(normalized["n_estimators"], [200])

        bad_candidate = "```json\n" + json.dumps({"reflection": "x", "configuration": {}, "extra": 1}) + "\n```"
        bad_extracted = agent.extract_json_hyperparams(bad_candidate)
        normalized_bad, reflection_bad, note_bad = agent.parse_reflection_configuration(bad_extracted)

        self.assertIsNone(normalized_bad)
        self.assertIsNone(reflection_bad)
        self.assertIn("exactly root keys", note_bad)

    def test_mixed_history_context_includes_best_and_recent_failures(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_file = Path(tmp_dir) / "results.tsv"

            with patch.object(agent, "RESULTS_FILE", results_file):
                agent.ensure_results_header()
                agent.append_result(1, 1.8, "kept", 0, FAST_TEST_MODEL, 0.0, "{}", "reflection one")
                agent.append_result(2, None, "crash", 0, FAST_TEST_MODEL, 0.0, "{}", "fatal_error=ValueError")
                agent.append_result(3, 1.3, "kept", 0, FAST_TEST_MODEL, 0.0, "{}", "reflection two")
                agent.append_result(4, None, "timeout", 0, FAST_TEST_MODEL, 0.0, "{}", "timeout_after=60.0s")
                agent.append_result(5, 2.2, "reverted", 0, FAST_TEST_MODEL, 0.0, "{}", "reflection three")

                context = agent.mixed_history_context(best_limit=3, failure_limit=2)

            self.assertIn("TOP 3 BEST-SCORING RUNS", context)
            self.assertIn("iter=3", context)
            self.assertIn("iter=1", context)
            self.assertIn("LAST 2 FAILED/NO-SCORE RUNS", context)
            self.assertIn("iter=4", context)
            self.assertIn("iter=2", context)

    def test_run_loop_fatal_stderr_halts_without_repair_attempt(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            target_file = temp_path / "train.py"
            backup_file = temp_path / "train_backup.py"
            results_file = temp_path / "results.tsv"
            data_dir = temp_path / "data"

            data_dir.mkdir(parents=True, exist_ok=True)
            target_file.write_text("print('placeholder train script')\n", encoding="utf-8")

            fatal_reflection = "The run history favors moderate depth, but this candidate may still expose missing env deps."
            first_generation_json = build_dual_output_candidate(
                fatal_reflection,
                {
                    "n_estimators": [100],
                    "max_depth": [5],
                    "learning_rate": [0.1],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "gamma": [0.0],
                    "reg_alpha": [0.0],
                    "reg_lambda": [1.0],
                },
            )

            with (
                patch.object(agent, "TARGET_FILE", target_file),
                patch.object(agent, "BACKUP_FILE", backup_file),
                patch.object(agent, "RESULTS_FILE", results_file),
                patch.object(agent, "DATA_DIR", data_dir),
                patch.object(agent, "Client", DummyClient),
                patch.object(agent, "verify_training_environment", autospec=True, return_value=(True, "ok")),
                patch.object(agent, "ask_ollama", autospec=True, return_value=first_generation_json) as mock_ask_ollama,
                patch.object(
                    agent,
                    "run_experiment",
                    autospec=True,
                    return_value=(1, "", "Traceback... ModuleNotFoundError: No module named numpy"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    agent.run_loop(
                        iterations=1,
                        temperature=0.0,
                        model=FAST_TEST_MODEL,
                        python_exe=sys.executable,
                        pause_s=0.0,
                        ollama_host=OLLAMA_HOST,
                    )

            mock_ask_ollama.assert_called_once()

            with results_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "crash")
            self.assertEqual(rows[0]["retries"], "0")
            self.assertIn("fatal_error=ModuleNotFoundError", rows[0]["note"])
            self.assertIn(fatal_reflection, rows[0]["note"])

    def test_run_loop_non_fatal_crash_still_uses_repair_path(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            target_file = temp_path / "train.py"
            backup_file = temp_path / "train_backup.py"
            results_file = temp_path / "results.tsv"
            data_dir = temp_path / "data"

            data_dir.mkdir(parents=True, exist_ok=True)
            target_file.write_text("print('placeholder train script')\n", encoding="utf-8")

            generation_json = build_dual_output_candidate(
                "The baseline is noisy; I will start with shallower trees and moderate learning rate.",
                {
                    "n_estimators": [120],
                    "max_depth": [4],
                    "learning_rate": [0.08],
                    "subsample": [0.85],
                    "colsample_bytree": [0.9],
                    "gamma": [0.1],
                    "reg_alpha": [0.0],
                    "reg_lambda": [1.2],
                },
            )
            repair_reflection = "The failure suggests brittle params, so I reduce variance with deeper but smoother settings."
            repair_json = build_dual_output_candidate(
                repair_reflection,
                {
                    "n_estimators": [180],
                    "max_depth": [6],
                    "learning_rate": [0.06],
                    "subsample": [0.9],
                    "colsample_bytree": [0.95],
                    "gamma": [0.0],
                    "reg_alpha": [0.0],
                    "reg_lambda": [1.0],
                },
            )

            with (
                patch.object(agent, "TARGET_FILE", target_file),
                patch.object(agent, "BACKUP_FILE", backup_file),
                patch.object(agent, "RESULTS_FILE", results_file),
                patch.object(agent, "DATA_DIR", data_dir),
                patch.object(agent, "Client", DummyClient),
                patch.object(agent, "verify_training_environment", autospec=True, return_value=(True, "ok")),
                patch.object(agent, "ask_ollama", autospec=True, side_effect=[generation_json, repair_json]) as mock_ask_ollama,
                patch.object(
                    agent,
                    "run_experiment",
                    autospec=True,
                    side_effect=[
                        (1, "", "ValueError: bad hyperparam"),
                        (0, "val_rmse: 1.234\n", ""),
                    ],
                ),
            ):
                agent.run_loop(
                    iterations=1,
                    temperature=0.0,
                    model=FAST_TEST_MODEL,
                    python_exe=sys.executable,
                    pause_s=0.0,
                    ollama_host=OLLAMA_HOST,
                )

            self.assertEqual(mock_ask_ollama.call_count, 2)

            with results_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "kept")
            self.assertEqual(rows[0]["retries"], "1")
            self.assertEqual(rows[0]["note"], repair_reflection)

    def test_single_iteration_keeps_better_candidate_and_logs_result(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            target_file = temp_path / "train.py"
            backup_file = temp_path / "train_backup.py"
            results_file = temp_path / "results.tsv"
            hyperparams_file = temp_path / "hyperparams.json"

            target_file.write_text("print('val_rmse: 9.00000')\n", encoding="utf-8")
            reflection = "Lower depth and moderate shrinkage should reduce overfit and improve validation RMSE."
            candidate_code = build_dual_output_candidate(reflection)

            with (
                patch.object(agent, "TARGET_FILE", target_file),
                patch.object(agent, "BACKUP_FILE", backup_file),
                patch.object(agent, "RESULTS_FILE", results_file),
                patch.object(agent, "DATA_DIR", temp_path),
                patch.object(agent, "Client", DummyClient),
                patch.object(agent, "verify_training_environment", autospec=True, return_value=(True, "ok")),
                patch.object(agent, "ask_ollama", autospec=True, return_value=candidate_code),
                patch.object(agent, "run_experiment", autospec=True, return_value=(0, "val_rmse: 1.50000\n", "")),
            ):
                agent.run_loop(
                    iterations=1,
                    temperature=0.0,
                    model=FAST_TEST_MODEL,
                    python_exe=sys.executable,
                    pause_s=0.0,
                    ollama_host=OLLAMA_HOST,
                )

            saved_hyperparams = json.loads(hyperparams_file.read_text(encoding="utf-8"))
            self.assertEqual(saved_hyperparams["n_estimators"], [200])

            with results_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "kept")
            self.assertEqual(rows[0]["score"], "1.500000")
            self.assertEqual(rows[0]["retries"], "0")
            self.assertEqual(rows[0]["note"], reflection)

    def test_single_iteration_reverted_still_persists_reflection(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            target_file = temp_path / "train.py"
            backup_file = temp_path / "train_backup.py"
            results_file = temp_path / "results.tsv"

            target_file.write_text("print('val_rmse: 9.00000')\n", encoding="utf-8")
            reflection = "The candidate is exploratory and may be worse, but it tests whether wider trees help."
            candidate_code = build_dual_output_candidate(reflection)

            with (
                patch.object(agent, "TARGET_FILE", target_file),
                patch.object(agent, "BACKUP_FILE", backup_file),
                patch.object(agent, "RESULTS_FILE", results_file),
                patch.object(agent, "DATA_DIR", temp_path),
                patch.object(agent, "Client", DummyClient),
                patch.object(agent, "verify_training_environment", autospec=True, return_value=(True, "ok")),
                patch.object(agent, "ask_ollama", autospec=True, return_value=candidate_code),
                patch.object(agent, "run_experiment", autospec=True, return_value=(0, "val_rmse: 4.50000\n", "")),
            ):
                agent.ensure_results_header()
                agent.append_result(0, 1.000000, "kept", 0, FAST_TEST_MODEL, 0.0, "{}", "historical best")
                agent.run_loop(
                    iterations=1,
                    temperature=0.0,
                    model=FAST_TEST_MODEL,
                    python_exe=sys.executable,
                    pause_s=0.0,
                    ollama_host=OLLAMA_HOST,
                )

            with results_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["status"], "reverted")
            self.assertEqual(rows[1]["note"], reflection)

    def test_run_loop_timeout_is_logged(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            target_file = temp_path / "train.py"
            backup_file = temp_path / "train_backup.py"
            results_file = temp_path / "results.tsv"

            target_file.write_text("print('placeholder train script')\n", encoding="utf-8")
            reflection = "I keep conservative values while probing timeout behavior in this controlled test."
            candidate_code = build_dual_output_candidate(reflection)

            with (
                patch.object(agent, "TARGET_FILE", target_file),
                patch.object(agent, "BACKUP_FILE", backup_file),
                patch.object(agent, "RESULTS_FILE", results_file),
                patch.object(agent, "DATA_DIR", temp_path),
                patch.object(agent, "Client", DummyClient),
                patch.object(agent, "verify_training_environment", autospec=True, return_value=(True, "ok")),
                patch.object(agent, "ask_ollama", autospec=True, return_value=candidate_code),
                patch.object(agent, "run_experiment", autospec=True, return_value=(124, "", "Training timed out after 60.0s", True)),
            ):
                agent.run_loop(
                    iterations=1,
                    temperature=0.0,
                    model=FAST_TEST_MODEL,
                    python_exe=sys.executable,
                    pause_s=0.0,
                    ollama_host=OLLAMA_HOST,
                )

            with results_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "timeout")
            self.assertIn(reflection, rows[0]["note"])


if __name__ == "__main__":
    unittest.main()
