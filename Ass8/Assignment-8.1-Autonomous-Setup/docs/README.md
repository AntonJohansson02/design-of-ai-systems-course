Autonomous Kaggle Agent Runbook (Local Qwen3.5:2B)

## Prerequisites
1. **Activate your virtual environment** (your prompt should look like `(.venv) PS ...`).
2. Install dependencies:
   ```powershell
   python -m pip install ollama
   ```
3. Pull the local model:
   ```powershell
   ollama pull qwen3.5:2b
   ```

## 1. Prepare Data (Run Once)
Run this from the **repo root** folder:
```powershell
python Ass8/Assignment-8.1-Autonomous-Setup/scripts/prepare.py
```

## 2. Check Baseline (Optional)
Make sure the base training script runs successfully:
```powershell
Push-Location Ass8/Assignment-8.1-Autonomous-Setup/data/prepared
python ../../src/autonomous_agent/train.py
Pop-Location
```
*(Expected output: one JSON line, e.g. `{"status":"ok","score":<number>,...}`)*

## 3. Run the Autonomous Agent
This will run the agent for 2 iterations to test the loop:
```powershell
python Ass8/Assignment-8.1-Autonomous-Setup/src/autonomous_agent/agent.py --iterations 2 --temperature 0.2 --python-exe python
```

**Run forever:**
```powershell
python Ass8/Assignment-8.1-Autonomous-Setup/src/autonomous_agent/agent.py --iterations 0 --temperature 0.2 --python-exe python
```

## Advanced Options

**Use a stronger model:**
```powershell
python Ass8/Assignment-8.1-Autonomous-Setup/src/autonomous_agent/agent.py --model qwen3.5:9b --iterations 20 --temperature 0.2 --python-exe python
```

**Run fast tests:**
```powershell
python -m unittest discover -s Ass8/Assignment-8.1-Autonomous-Setup/tests -p "test_*.py" -v
```

---

### Failure handling and guardrails
- Crashes trigger self-healing prompts up to 3 attempts.
- Training runs have a strict timeout (`--train-timeout`, default 60s).
- Oversized or invalid switchboard JSON candidates are rejected before training.
- The agent only keeps candidates that improve the best RMSE score.

