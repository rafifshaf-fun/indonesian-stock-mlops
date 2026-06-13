# Contributing to Indonesian Stock MLOps

Thanks for your interest in contributing! This document explains how to set up your environment, make changes, and submit them.

## Development Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/indonesian-stock-mlops.git
cd indonesian-stock-mlops

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dev dependencies
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 2. Start supporting services

```bash
# Start MLflow, Prometheus, and Grafana (API runs locally for dev)
docker compose up -d mlflow prometheus grafana

# Run the API locally (not in Docker) for faster iteration
python src/serve.py
```

### 3. Verify setup

```bash
python cli.py status     # Should show API running
python cli.py predict BBCA.JK  # Should return prediction
```

## Project Architecture

```
src/features/     Modular feature engineering — each submodule handles one category
src/serve.py      FastAPI server — prediction endpoints + sentiment overlay
src/train.py      XGBoost training — per-ticker models + Optuna tuning
src/backtest.py   Walk-forward backtesting engine
src/config.py     Single source of truth — tickers, sectors, feature flags, paths
cli.py            Unified CLI — predict, backtest, sentiment, train, serve
```

### Adding a new feature category

1. Create a new module in `src/features/` (e.g., `src/features/my_features.py`)
2. Add a feature flag in `src/config.py` → `FEATURE_FLAGS`
3. Wire into `engineer_features_for_ticker()` in `src/features/pipeline.py`
4. Wire into `build_inference_features()` in `src/serve.py`
5. Export from `src/features/__init__.py`

### Adding a new CLI command

1. Add a function `cmd_mycommand(args)` to `cli.py`
2. Add a subparser in `main()`
3. Register in the `commands` dict

## Code Style

- **Python 3.11+** with type hints where helpful
- **4 spaces** for indentation (no tabs)
- **Docstrings** for all public functions (Google-style preferred)
- **Feature flags** for any optional feature (see `config.py`)
- **Graceful fallbacks** — all external API calls should have try/except with safe defaults

## Testing

```bash
# Run backtest to validate models
python cli.py backtest

# Test predictions end-to-end
python cli.py predict BBCA.JK BBRI.JK

# Test local mode (no API server needed)
python cli.py predict BBCA.JK --local

# Check sentiment
python cli.py sentiment BBCA.JK
```

## Model Workflow

1. **Fetch data**: `python src/ingest.py`
2. **Build features**: Run via `build_feature_set()` in `src/features/pipeline.py`
3. **Train models**: `python src/train.py` or `python cli.py train`
4. **Build index**: `python scripts/build_model_index.py`
5. **Backtest**: `python src/backtest.py`
6. **Deploy**: `docker compose up -d api`

## Docker Notes

- The API container bind-mounts `.` → `/app` for live code changes
- Use `127.0.0.1` (not `localhost`) on Docker Desktop Windows
- `requirements-docker.txt` is the slim version (no DVC)
- Model index must be rebuilt after retraining: `python scripts/build_model_index.py`

## Submitting Changes

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feat/my-feature`
3. Make your changes and test locally
4. Run `python cli.py backtest` to verify models still work
5. Commit with a clear message: `git commit -m "feat: add my feature"`
6. Push and open a **Pull Request**

### Commit convention

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring |
| `docs:` | Documentation |
| `perf:` | Performance improvement |

## Questions?

Open an issue or start a discussion. Happy to help!
