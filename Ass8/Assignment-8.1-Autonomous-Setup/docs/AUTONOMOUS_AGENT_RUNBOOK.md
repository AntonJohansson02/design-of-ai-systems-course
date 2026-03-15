Autonomous Kaggle Agent Runbook (Local Qwen3.5:2B)

Prerequisites
- Ollama is installed and running on localhost:11434.
- Python package dependency installed:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe -m pip install ollama
- Model pulled locally:
ollama pull qwen3.5:2b

What this setup contains
- scripts/prepare.py: One-time static preprocessing into NumPy arrays.
- src/autonomous_agent/train.py: Agent-editable training script under 50 lines.
- src/autonomous_agent/agent.py: Closed-loop orchestrator with keep-or-revert logic.
- outputs/results.tsv: Append-only experiment ledger.

One-time preparation
1. Open PowerShell in this folder.
2. Run:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe scripts/prepare.py

Default data source
- scripts/prepare.py reads train.csv and test.csv from ../Assignment-8.0-Your-mini-project by default.
- Override with:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe scripts/prepare.py --source-data-dir ../Assignment-8.0-Your-mini-project --output-dir data/prepared

This creates:
- data/prepared/X_train.npy, data/prepared/y_train.npy
- data/prepared/X_val.npy, data/prepared/y_val.npy
- data/prepared/X_test.npy, data/prepared/test_ids.npy
- data/prepared/feature_names.npy, data/prepared/dataset_metadata.json

Manual baseline check
Run:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe src/autonomous_agent/train.py

Expected output format:
val_rmse: 24446.97070

Fast tests (about 1-3 seconds)
Run:
C:/Users/anton/OneDrive/Skrivbord/design-of-ai-systems-course/.venv/Scripts/python.exe -m unittest discover -s Ass8/Assignment-8.1-Autonomous-Setup/tests -p "test_*.py" -v

What this validates quickly
- Real Ollama Python library call via chat(..., think=True) against qwen3.5:2b.
- One deterministic end-to-end agent loop iteration with mocked code generation and real subprocess execution.

Quick Ollama connectivity check (optional)
Run:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe -c "from ollama import chat; messages=[{'role':'user','content':'What is 10 + 23?'}]; response=chat('qwen3.5:2b', messages=messages, think=True); print('Thinking:\n========\n\n' + (response.message.thinking or '')); print('\nResponse:\n========\n\n' + (response.message.content or ''))"

Expected response section includes:
33

Run the autonomous loop
Run in a new terminal:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe src/autonomous_agent/agent.py --iterations 0 --temperature 0.2 --python-exe C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe

Optional stronger model (slower):
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe src/autonomous_agent/agent.py --model qwen3.5:9b --iterations 20 --temperature 0.2 --python-exe C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe

Optional host override
- If Ollama is exposed on a non-default endpoint:
C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe src/autonomous_agent/agent.py --ollama-host http://localhost:11434 --iterations 20 --temperature 0.2 --python-exe C:/Users/anton/AppData/Local/Programs/Python/Python313/python.exe

Notes
- --iterations 0 means run forever.
- Use --iterations 20 for a finite test batch.
- Use --pause 1.5 to add delay between iterations.
- Default model is qwen3.5:2b for speed. Use --model qwen3.5:9b for higher-capacity runs.
- The agent only keeps changes that improve best RMSE.
- The agent asks Ollama via chat(..., think=True) and uses the final response content from that same call.

Failure handling and guardrails
- Regex extraction strips markdown wrappers and extra prose from model responses.
- Crashes trigger self-healing prompts up to 3 attempts.
- If all repairs fail, train.py is reverted to last known good copy.
- If val_rmse is not parseable, candidate code is reverted.

results.tsv schema
- timestamp_utc
- iteration
- score
- status (kept, reverted, crash, parse_fail, invalid, llm_error)
- retries
- model
- temperature
- code_hash
- note

