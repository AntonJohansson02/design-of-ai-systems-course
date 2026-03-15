from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

OLLAMA_HOST = "http://localhost:11434"
FAST_TEST_MODEL = "qwen3.5:2b"


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

    def test_single_iteration_keeps_better_candidate_and_logs_result(self):
        agent = load_agent_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            target_file = temp_path / "train.py"
            backup_file = temp_path / "train_backup.py"
            results_file = temp_path / "results.tsv"

            target_file.write_text("print('val_rmse: 9.00000')\n", encoding="utf-8")
            candidate_code = "print('val_rmse: 1.50000')\n"

            with (
                patch.object(agent, "TARGET_FILE", target_file),
                patch.object(agent, "BACKUP_FILE", backup_file),
                patch.object(agent, "RESULTS_FILE", results_file),
                patch.object(agent, "DATA_DIR", temp_path),
                patch.object(agent, "Client", DummyClient),
                patch.object(agent, "ask_ollama", autospec=True, return_value=candidate_code),
            ):
                agent.run_loop(
                    iterations=1,
                    temperature=0.0,
                    model=FAST_TEST_MODEL,
                    python_exe=sys.executable,
                    pause_s=0.0,
                    ollama_host=OLLAMA_HOST,
                )

            final_train = target_file.read_text(encoding="utf-8")
            self.assertIn("val_rmse: 1.50000", final_train)

            with results_file.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "kept")
            self.assertEqual(rows[0]["score"], "1.500000")
            self.assertEqual(rows[0]["retries"], "0")


if __name__ == "__main__":
    unittest.main()
