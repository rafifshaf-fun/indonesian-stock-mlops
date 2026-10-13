# Project instructions

Indonesian stock (IDX/LQ45) BUY/SELL prediction: yfinance + FRED features →
per-ticker XGBoost/LightGBM models → FastAPI serving → Prometheus/Grafana.
Python 3.11, local venv at `venv/`.

## Commands

- Test: `.\venv\Scripts\python.exe -m pytest` — the suite under `tests/`, configured in
  `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`, and
  `filterwarnings = ["error"]`, which is deliberate: a stray warning is a failure, so keep the
  output pristine). Six modules still call `warnings.filterwarnings("ignore")` at import time
  (`cli.py`, `predict.py`, `src/train.py`, `src/backtest.py`, `src/ingest.py`,
  `src/news_sentiment.py`); those suppress warnings raised during import/collection, not in tests.
- Lint / Types / Build: none configured. `ruff`, `black` and `pre-commit` are installed and pinned
  in `requirements-dev.txt` but have no configuration, and there is no type checker or build step.
- Full stack: `docker compose up -d`, then `python cli.py status`.
- API only, no Docker (faster dev loop): `python src/serve.py`.
- CLI entry point: `python cli.py {predict,backtest,sentiment,train,serve,status,list}`.
- Daily pipeline: `python src/ingest.py` → `python scripts/ci_features.py`
  → `python scripts/seed_metrics.py`.
- Weekly retrain: the daily steps, then
  `python src/train.py --parallel --features data/processed/features.csv`
  → `python scripts/build_model_index.py`
  → `docker compose up -d --force-recreate api` → `python scripts/seed_metrics.py`.
- Validate the compose file with `docker compose config --quiet`.

## Layout

- `src/config.py` — the single source of truth: `TICKERS` (45), `SECTORS` (7),
  data paths, `FEATURE_FLAGS`, `TRAINING_CONFIG`, `CACHE_CONFIG`. Importing it
  also installs the root logging config, so import it before logging.
- `src/features/` — feature package. `pipeline.py` is the only orchestrator
  (`build_feature_set`, `engineer_features_for_ticker`); the other modules are
  leaf feature families.
- `src/serve.py` (FastAPI app), `src/train.py` (training), `src/backtest.py`
  (walk-forward), `src/ingest.py` (yfinance fetch), `src/news_sentiment.py`.
- `mlruns/1/models/` — where prediction and backtest code looks for models. The
  path segment `1` is hardcoded in `cli.py`, `predict.py`, `src/serve.py`,
  `src/backtest.py` and three scripts under `scripts/`. Do not renumber it.
- `mlruns/1/models/model_index.json` — the only file under `mlruns/` tracked in
  git; maps ticker → model folder. Every prediction resolves models through it.
- `scripts/` — operational one-offs run by hand or by CI; not imported by `src/`.
- `monitoring/` — Prometheus scrape config plus provisioned Grafana
  datasources and dashboards.
- `app/` — gitignored, holds only a stray `mlflow.db`. Not application code.
- `data/intraday/`, `data/sentiment/`, `data/prediction_cache.db` are caches;
  `data/raw/stocks.csv` and `data/processed/features.csv` are DVC-tracked.

## Conventions

- Adding a feature category takes four edits, not one: a module in
  `src/features/`, a flag in `config.FEATURE_FLAGS`, a call in
  `pipeline.engineer_features_for_ticker`, and a call in
  `serve.build_inference_features`, plus an export from `src/features/__init__.py`.
  A feature flag by itself does nothing.
- `build_feature_set(mode=...)`: `ci` skips `news_sentiment`, `volume_profile`
  and `google_trends` and writes the CSV; `full` writes the parquet.
  `scripts/ci_features.py` always uses `mode='ci'`.
- Every external API call is wrapped in try/except with a safe default; a
  sentiment failure must never block a prediction (`serve.py` sentiment overlay)
  — but the fallback must be visible. `PredictResponse.degraded` names every
  component that failed (`["sentiment"]`), the overlay logs at `warning`, and
  `compute_volume_profile_features`/`build_features` warn when a family is
  omitted. A safe default indistinguishable from a real value is the defect.
- `/predict/batch` raises `HTTPException(502, detail=...)` for a ticker that
  cannot be predicted. It used to answer `prediction: 0` (SELL) with
  `probability_up: 0.0` under HTTP 200, which a caller could trade on.
- `drift_warning` and `drift_score` are on every response and are **not
  computed** — hardcoded `false`/`0.0`, with `_compute_psi` never called. They
  are kept for compatibility only; wiring PSI needs a reference distribution.
- `TICKERS` and `SECTORS` must stay in sync — the universe is LQ45 and changes.
- Commit style is prefixed: `feat:`, `fix:`, `refactor:`, `docs:`, `perf:`.

## Workflow notes

- **Training ignores freshly rebuilt features by default.** `train.py` defaults
  to `data/processed/features.parquet` and only falls back to the CSV when the
  parquet is *absent*. The daily/CI feature step writes `features.csv`, and
  `start.bat` calls `python src/train.py` with no `--features`, so it retrains on
  the older parquet. Pass `--features data/processed/features.csv`, which is what
  CI does.
- **The model index maps no tickers.** `mlruns/1/models/model_index.json` in git has
  `mapped_tickers: 0` while 45 model folders exist locally. Selection is now single-path
  (`src/model_store.py`): an unmapped ticker raises `ModelNotFound`, which the API surfaces as a
  404 and a startup `warning` names the remedy. The "first valid model folder" substitute — which
  silently served another ticker's model — is deleted, along with the backtester's reverse-sorted
  scan that made the API and the backtest disagree. Rebuild with
  `python scripts/build_model_index.py`, or let training write it (training is now its single
  writer). Until the index is populated, every prediction fails loudly by design.
- Each model carries a `feature_schema.json` beside its `MLmodel`
  (`mlruns/1/models/<model_id>/artifacts/`), written by training and enforced at load through
  `features.schema.align_to_schema`. A model whose schema names a column the serving builder
  cannot produce fails with `FeatureSchemaError` rather than padding it with `NaN`. Models
  trained before this contract existed are loaded by falling back to the model's own
  `feature_names_in_`, with a `warning` naming the ticker.
- Serving builds features by calling the same I/O-free `features.pipeline.build_features` that
  training uses, so the two paths cannot drift. The `vp_*`, `idn_*` and `google_trend` families
  are deliberately absent at inference — serving cannot produce them without per-request network
  I/O — and are excluded from the feature contract. Because `build_features` now warns whenever a
  family's flag is on and its argument is `None`, serving logs those deliberate omissions at
  `warning` once per request; that is expected, not a fault.
- Rebuilding the index is mandatory after every retrain, and the API loads it
  once at startup — restart the container (`--force-recreate api`) to pick it up.
- Training all 45 tickers takes ~15-20 min with `--parallel`. `--tune` covers only
  the top-10 tickers and is far slower.
- `--no-shap`, `--no-expand-cv` and `--no-blend` are silently ignored when
  `--parallel` is used: the parallel branch forwards only ticker, `--tune` and
  `--features`, so `train()` falls back to its all-on defaults. Combine those
  flags only on the sequential path.
- CI (`.github/workflows/ci.yml`) is read-only: it declares `permissions: contents: read`, has no
  `git push`, and no longer commits `mlflow.db` or `mlruns/` back to `main`. Actions are pinned by
  commit SHA. `mlflow.db` is untracked, so experiment history is local or Docker-only.
- The `api` container bind-mounts the repo root to `/app`, so it executes host
  code and reads host `data/` and `mlruns/`. Code changes need no image rebuild;
  model-index changes need a container restart.
- `pip install -r requirements.txt` does **not** install `vaderSentiment` or
  `nltk`; the image uses `requirements-docker.txt`, which does, and CI installs
  both ad hoc. Sentiment degrades to a 0.0 score in a plain venv — visible now
  as `degraded: ["sentiment"]` and a `warning`, not only in the score.
- `.env` (`FRED_API_KEY`, `NEWSAPI_KEY`) is gitignored; `.env.example` is
  committed, exempted from the `.env.*` rule by the `!.env.example` negation, and
  documents both keys with empty values. CI reads `FRED_API_KEY` from the
  Actions secret of the same name. A missing key degrades features to NaN/0
  instead of failing.
- `start.bat`'s menu works: the jump that skipped options 7 (Daily) and 8 (Weekly) is now
  conditional, and the unescaped parentheses that corrupted the `if (...)` blocks are escaped.
  In batch, an unescaped `(` or `)` inside a parenthesised block terminates that block early —
  worth remembering before editing this file.
- DVC tracks `data/raw/stocks.csv` and `data/processed/features.csv`, but the
  configured remote (`../UsersLAdmininstratordvc-storage`) is a mangled local
  path, so `dvc pull` will not resolve until the remote is re-added.
- Use `127.0.0.1`, not `localhost`, for API calls on Docker Desktop Windows.
- No browser UI exists in this repository. The only web surfaces are the MLflow,
  Prometheus and Grafana containers. Do not invoke `ui-ux-pro-max`,
  `apple-design`, or `design-principles` here, and browser verification does not
  apply. Verify the API with `curl`/`python cli.py status` instead.
