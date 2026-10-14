# Signal Correctness Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the BUY/SELL signal mean what it claims — for the right stock, from features the model was actually trained on — and prove it with tests, before any dashboard or data-layer revamp begins.

**Architecture:** The training path and the serving path currently build features with two separate, divergent implementations, and the model is trained on standardised features while serving feeds raw ones. This plan collapses both paths onto a single pure feature function, enforces one explicit feature schema per model, removes standardisation entirely (gradient-boosted trees are invariant to monotone per-feature transforms, so it bought nothing and cost correctness), replaces the two silent model-selection fallbacks with one that raises, and has training write the model index as its single writer.

**Tech Stack:** Python 3.11.6, pandas, scikit-learn 1.9.0, XGBoost 3.2.0, LightGBM 4.6.0, MLflow 3.13.0, FastAPI, pytest.

**Spec:** This plan implements the findings of the repository audit of 2026-10-13. Finding IDs `F1`–`F12` below refer to that audit's ranked table. **All 45 existing models are scrap** and must be retrained: they were fitted on standardised features and the scaler was never persisted, so the transform is unrecoverable.

---

## Global Constraints

- Python 3.11.6; the working interpreter is `venv\Scripts\python.exe`.
- Do not change the ticker universe, the 45-ticker count, or the `mlruns/1/models` location during this plan.
- Do not commit `mlflow.db`, `mlruns/**/artifacts/`, `.pyc` files, `.env`, or any credential.
- No secret may appear in a tracked file; all credentials come from environment variables or GitHub Secrets.
- Keep MLflow as the experiment store. Do not replace it.
- All timestamps stored in UTC; all times rendered in `Asia/Jakarta` (WIB, UTC+7, no DST).
- Every external call (yfinance, FRED, NewsAPI, BI) stays wrapped with a safe default, but every *degradation* must be visible in the response or the log at `warning` or above.
- A missing or mismatched model must raise. Silent substitution is forbidden anywhere in the prediction path.
- No behavioural claim may be documented in `README.md` unless a test in `tests/` exercises it.

## Review Focus

The audit's implied-but-untested inputs, most likely to bite first. Each line's test lives in the task named beside it.

1. **A ticker in `TICKERS` with no raw data row** — expect a clear `ModelNotFound`/`InsufficientData` error, never a fallback model serving another stock's opinion. (Task 1.6)
2. **A model whose saved schema differs from the currently computed feature frame** (after any future feature change) — expect a hard failure at load or predict time, never NaN-padded predictions. (Task 1.4)
3. **A stale ingest**: `data/raw/stocks.csv` or `ihsg.csv` older than the requested trading day — expect an explicit staleness signal in the response, not silently stale features. (Task 1.3)
4. **Column order drift** between the saved schema and the computed frame — expect exact-order enforcement, since positional alignment silently permutes values. (Task 1.8)
5. **A non-trading day or market holiday** — expect no signal and no crash. (Task 1.10)
6. **The sentiment API being down or `NEWSAPI_KEY` missing** — expect an explicit degraded flag, never `sentiment_score: 0.0` indistinguishable from neutral. (Task 1.9)

---

## Phase 0 — Stop the bleeding

### Task 0.1: Remove the committed credential

**Files:**
- Modify: `.github/workflows/ci.yml:35-38`
- Create: `.env.example`
- Modify: `AGENTS.md` (Workflow notes)

**Interfaces:**
- Consumes: nothing.
- Produces: the CI step reads `FRED_API_KEY` from the Actions secret of the same name; `.env.example` documents `FRED_API_KEY` and `NEWSAPI_KEY` with empty values.

- [ ] **Step 1: Confirm the exposure and record it**

Run: `git log --all -S"FRED_API_KEY: " --oneline -- .github/workflows/ci.yml`
Expected: matches the commits that introduced the literal. Record the count in the commit message; do not paste the value anywhere.

- [ ] **Step 2: Replace the literal with a secret reference**

In `.github/workflows/ci.yml`, change the `FRED_API_KEY:` line in the feature-engineering step to `FRED_API_KEY: ${{ secrets.FRED_API_KEY }}`.

- [ ] **Step 3: Add `.env.example`**

Create `.env.example` containing `FRED_API_KEY=` and `NEWSAPI_KEY=` with empty values and a one-line comment each.

- [ ] **Step 4: Verify no literal remains in the working tree**

Run: `git grep -n -E "[A-Za-z0-9]{32}" -- .github/`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml .env.example AGENTS.md
git commit -m "fix(security): move FRED key out of CI into a secret; add .env.example

The key is present in every revision since 0675166 and the repository is
public, so history rewriting does not remove the exposure: the key must be
rotated in the FRED account. That rotation is a human action, not a commit."
```

> **Human action required, blocking:** rotate the FRED key in the FRED account and store the new value as the `FRED_API_KEY` Actions secret. The code change alone does not fix the exposure.

### Task 0.2: Stop CI from writing to the repository

**Files:**
- Modify: `.github/workflows/ci.yml` (permissions, both `uses:` lines, the commit step)

**Interfaces:**
- Consumes: Task 0.1's secret reference.
- Produces: a read-only CI topology — no `contents: write`, no `git push`, actions pinned by SHA.

- [ ] **Step 1: Remove the write permission and the commit-to-main step**

Delete `permissions: contents: write` and the entire "Commit metrics" step. CI's job is to verify, not to publish. `mlflow.db` must stop being committed; if experiment history must be shared, it belongs in a database, not in git.

- [ ] **Step 2: Pin both actions to commit SHAs**

Replace `actions/checkout@v4` and `actions/setup-python@v5` with their pinned `@<40-hex-sha>` forms, keeping the version in a trailing comment.

- [ ] **Step 3: Verify the workflow still parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Verify no write path remains**

Run: `git grep -n -E "contents: write|git push|git add -f" -- .github/`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "fix(ci): make CI read-only and pin actions to SHAs"
```

### Task 0.3: Pin dependencies and fix the manifest

**Files:**
- Modify: `requirements.txt`, `requirements-docker.txt`, `requirements-dev.txt`, `Dockerfile:11-12`

**Interfaces:**
- Consumes: nothing.
- Produces: `requirements.txt` with `==` pins for everything installed today; the four undeclared imports (`requests`, `PyYAML`, `prometheus_client`, `joblib`) declared; `Dockerfile` referencing `requirements-docker.txt` by its own name.

- [ ] **Step 1: Freeze what is installed today**

Run: `.\venv\Scripts\python.exe -m pip freeze > requirements.lock.txt`
Expected: a file with `==` pins for the ~150 resolved packages. This is the known-good artefact; it is the reproducibility baseline.

- [ ] **Step 2: Pin the direct dependencies**

Rewrite `requirements.txt` with `==` pins matching the installed versions from `pip freeze`, and add the four missing declarations. Remove `dvc` (unused, remote unreachable), `pandas-datareader`, `tenacity` and `pyarrow` only if Step 3 confirms nothing imports them.

- [ ] **Step 3: Verify no import is undeclared**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_manifest.py -v` after Task 1.1 creates the harness; until then run the equivalent check inline:
`.\venv\Scripts\python.exe -c "import requests, yaml, prometheus_client, joblib; print('ok')"`
Expected: `ok` — and the same four names present in `requirements.txt`.

- [ ] **Step 4: Fix the Dockerfile to reference the docker manifest honestly**

Replace `COPY requirements-docker.txt requirements.txt` + `RUN pip install --no-cache-dir -r requirements.txt` with:
`COPY requirements-docker.txt ./` + `RUN pip install --no-cache-dir -r requirements-docker.txt`.
Behaviour is unchanged (the rename already installed the docker manifest), but the file no longer lies to anyone reading the running container.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements-docker.txt requirements.lock.txt Dockerfile
git commit -m "chore: pin dependencies, declare missing imports, fix Dockerfile manifest"
```

### Task 0.4: Repository hygiene

**Files:**
- Modify: `.gitignore`, `README.md:277`, `start.bat:112-168`, `src/config.py:22-28`, `src/config.py:56-74`
- Remove from tracking: `src/__pycache__/*.pyc`

**Interfaces:**
- Consumes: nothing.
- Produces: a `.gitignore` whose every line matches; `start.bat` options 7 and 8 reachable; `SECTORS` covering all 45 tickers; `TICKER_SEARCH_TERMS` covering all 45 tickers.

- [ ] **Step 1: Rewrite the malformed ignore line as ASCII**

Rewrite `.gitignore` as UTF-8/ASCII from scratch, preserving every existing rule. The last line is currently UTF-16LE and matches nothing.

- [ ] **Step 2: Untrack the committed bytecode**

```bash
git rm -r --cached src/__pycache__
```

- [ ] **Step 3: Verify every ignore rule matches something**

Run: `git check-ignore -v scripts/_diag.py .env mlflow.db`
Expected: three matching lines, one per path, each naming a rule in `.gitignore`.

- [ ] **Step 4: Fix the unreachable launcher options**

In `start.bat`, move the `goto END` currently at line 122 to after the option-8 block (i.e. immediately before the `:: OPTIONS 7 & 8` section becomes reachable, and after it, at the existing `:FAIL` label boundary). Verify by running `start.bat` and choosing `7`, then confirming the "DAILY PIPELINE COMPLETE" output appears.

- [ ] **Step 5: Fix the config drift and the broken README command**

Add `ADMR.JK` to `energy_mining` in `SECTORS`; add `INCO.JK`, `ITMG.JK`, `SIDO.JK` to `TICKER_SEARCH_TERMS`; change `README.md:277` from `python cli.py train --parallel --no-shap` to `python src/train.py --parallel --no-shap` (the CLI has no such flag).

- [ ] **Step 6: Verify config coverage**

Run: `.\venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from config import TICKERS, SECTORS, TICKER_SEARCH_TERMS; k=set(t for v in SECTORS.values() for t in v); print('no sector:', sorted(set(TICKERS)-k)); print('no search term:', sorted(set(TICKERS)-set(TICKER_SEARCH_TERMS)))"`
Expected: `no sector: []` and `no search term: []`

- [ ] **Step 7: Commit**

```bash
git add .gitignore README.md start.bat src/config.py
git commit -m "chore: fix gitignore encoding, launcher control flow, config drift, README command"
```

---

## Phase 1 — Make it correct

### Task 1.1: Test harness

**Files:**
- Create: `pyproject.toml`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`
- Modify: `requirements-dev.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `pytest` runnable from the repo root; fixtures `raw_ohlcv` (a synthetic 3-ticker, 300-row yfinance-style MultiIndex frame), `ihsg_frame`, `fundamentals_dict`, `tmp_model_dir`.

- [ ] **Step 1: Install the dev tools that were only ever declared**

Run: `.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt`
Expected: pytest, pytest-asyncio, httpx, black, ruff, pre-commit, ipykernel all installed.

- [ ] **Step 2: Add pytest configuration**

Create `pyproject.toml` with `[tool.pytest.ini_options]` setting `testpaths = ["tests"]`, `pythonpath = ["src"]`, and `filterwarnings = ["error"]`. The last one matters: the codebase currently silences every warning globally, and those warnings are how the column-alignment and NaN problems would announce themselves.

- [ ] **Step 3: Write the failing test**

```python
# tests/test_smoke.py
def test_imports_resolve():
    import config, features  # noqa: F401
    assert True

def test_fixture_shape(raw_ohlcv):
    assert raw_ohlcv.shape[1] == 3 * 5  # 3 tickers x OHLCV
```

- [ ] **Step 4: Run it and confirm the harness works**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_smoke.py -v`
Expected: PASS. If `filterwarnings = ["error"]` turns existing imports into failures, fix the offending `warnings.filterwarnings("ignore")` call in the module being imported rather than relaxing the config — that suppression is a finding, not a fixture.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/ requirements-dev.txt
git commit -m "test: add pytest harness and fixtures"
```

### Task 1.2: One pure feature builder

**Files:**
- Modify: `src/features/pipeline.py`
- Test: `tests/test_feature_builder.py`

**Interfaces:**
- Consumes: Task 1.1 fixtures.
- Produces: `build_features(df, ticker, *, fundamentals, usdidr, fred_macro, bi_rate, ihsg=None, all_ticker_data=None, news_sentiment=0.0, google_trend=0.0, vp_features=None, idx_fundamentals=None, mode="full", with_target=False) -> pd.DataFrame | None`. Performs **no I/O and no network calls**. `engineer_features_for_ticker` keeps its exact existing signature and becomes a thin wrapper that fetches, then calls `build_features(..., with_target=True)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_builder.py
def test_build_features_does_no_io(raw_ohlcv, ihsg_frame, fundamentals_dict, monkeypatch):
    import features.pipeline as p

    def explode(*a, **k):
        raise AssertionError("build_features must not fetch")

    monkeypatch.setattr(p, "fetch_fundamentals", explode)
    monkeypatch.setattr(p, "fetch_news_sentiment", explode)
    monkeypatch.setattr(p, "fetch_idx_fundamentals", explode)
    monkeypatch.setattr(p, "fetch_google_trends", explode)

    out = p.build_features(
        raw_ohlcv, "BBCA.JK",
        fundamentals=fundamentals_dict, usdidr=None, fred_macro=None, bi_rate=None,
        ihsg=ihsg_frame, all_ticker_data={"BBCA.JK": raw_ohlcv["BBCA.JK"].dropna()},
        news_sentiment=0.2, google_trend=0.0, vp_features={}, idx_fundamentals={},
        mode="full", with_target=True,
    )
    assert out is not None and len(out) > 50
    assert "target" in out.columns
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_feature_builder.py -v`
Expected: FAIL with `AttributeError: module 'features.pipeline' has no attribute 'build_features'`.

- [ ] **Step 3: Extract `build_features` from `engineer_features_for_ticker`**

Move the body of `engineer_features_for_ticker` (lines 113-166) into `build_features`, replacing each fetch with the matching keyword argument, and gating `compute_target` and the `ticker` column behind `with_target`. Rewrite `engineer_features_for_ticker` to perform the existing fetches and delegate. Keep the `mode != "ci"` gating for `news_sentiment`, `volume_profile` and `google_trend` exactly as-is.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_feature_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Verify the wrapper is behaviour-preserving**

Run: `.\venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import features.pipeline as p; print(p.engineer_features_for_ticker.__doc__ is not None)"`
Expected: `True` — the public signature is unchanged for existing callers.

- [ ] **Step 6: Commit**

```bash
git add src/features/pipeline.py tests/test_feature_builder.py
git commit -m "refactor: extract pure build_features from engineer_features_for_ticker"
```

### Task 1.3: Serving uses the same builder (F3, F5)

**Files:**
- Modify: `src/serve.py:287-356`
- Test: `tests/test_serve_parity.py`

**Interfaces:**
- Consumes: `build_features` from Task 1.2.
- Produces: `build_inference_features(df, ticker, fundamentals) -> pd.DataFrame` delegating to `build_features` with universe data loaded from disk; a module-level `load_universe_context() -> tuple[pd.DataFrame, pd.DataFrame, dict]` returning `(raw_ohlcv, ihsg, all_ticker_data)`, cached at startup; and `universe_age_days() -> int`. **The 32-column placeholder block is deleted.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serve_parity.py
def test_no_placeholder_nan_columns(raw_ohlcv, ihsg_frame, fundamentals_dict, monkeypatch):
    import serve
    monkeypatch.setattr(serve, "load_universe_context",
                        lambda: (raw_ohlcv, ihsg_frame, {"BBCA.JK": raw_ohlcv["BBCA.JK"].dropna()}))
    out = serve.build_inference_features(raw_ohlcv["BBCA.JK"].dropna(), "BBCA.JK", fundamentals_dict)
    all_nan = [c for c in out.columns if out[c].isna().all()]
    assert all_nan == [], f"features still faked at inference: {all_nan}"

def test_universe_staleness_is_reported(tmp_path, monkeypatch):
    import serve
    # fixture writes a stocks.csv with a 10-day-old mtime
    assert serve.universe_age_days() >= 10
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_serve_parity.py -v`
Expected: FAIL — `all_nan` contains `correlation_20d`, `sector_relative_strength`, `ihsg_return_5d`, `relative_strength_1d`, `sector_return_1d`, `beta_20d` and the `idn_*`/`vp_*` groups.

- [ ] **Step 3: Implement universe loading and delegate**

Add `load_universe_context()` reading `data/raw/stocks.csv` via `pipeline.load_data` and `data/raw/ihsg.csv` via `market.load_ihsg_data`, building `all_ticker_data` per ticker. Add `universe_age_days()` from the file mtime. Rewrite `build_inference_features` to call `build_features(..., with_target=False)` with values from the daily caches. Delete the three placeholder loops (lines 333-352).

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_serve_parity.py -v`
Expected: PASS, both tests.

- [ ] **Step 5: Commit**

```bash
git add src/serve.py tests/test_serve_parity.py
git commit -m "fix(serve): build inference features with the training builder (F3, F5)"
```

### Task 1.4: One explicit feature schema (F5)

**Files:**
- Create: `src/features/schema.py`
- Modify: `src/train.py:229-300` (schema persistence only — **not** `src/serve.py`; see Step 5)
- Test: `tests/test_feature_schema.py`

**Interfaces:**
- Consumes: Task 1.2's builder.
- Produces: `FeatureSchemaError(RuntimeError)`; `save_feature_schema(columns: list[str], path: str | Path) -> None`; `load_feature_schema(path: str | Path) -> list[str]`; `align_to_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame` which raises `FeatureSchemaError` listing every missing or extra column and never pads. Schema is persisted beside each model as `feature_schema.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_schema.py
import pandas as pd, pytest
from features.schema import save_feature_schema, load_feature_schema, align_to_schema, FeatureSchemaError

def test_roundtrip(tmp_path):
    p = tmp_path / "feature_schema.json"
    save_feature_schema(["b", "a"], p)
    assert load_feature_schema(p) == ["b", "a"]  # order preserved

def test_missing_column_raises_not_pads(tmp_path):
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(FeatureSchemaError, match="b"):
        align_to_schema(df, ["a", "b"])

def test_reordered_frame_is_reordered_not_permuted():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = align_to_schema(df, ["b", "a"])
    assert list(out.columns) == ["b", "a"]
    assert out["b"].iloc[0] == 2.0
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_feature_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'features.schema'`.

- [ ] **Step 3: Implement `src/features/schema.py`**

Implement the four names above. `align_to_schema` must reorder by name and raise on any mismatch; it must never insert `NaN`.

- [ ] **Step 4: Use one global feature selection and persist the schema**

In `src/train.py`, replace the per-ticker `_select_features_shap` call with a single selection computed once over the pooled frame, then train every ticker on that same ordered list. Write `feature_schema.json` into each logged model's directory before `log_model`.

- [ ] **Step 5: Do NOT wire enforcement here (deferred to Task 1.6)**

This task originally wired enforcement into `src/serve.py`. Task 1.6 moves all model loading into `src/model_store.py`, so wiring it here would be thrown away. **Do not modify `src/serve.py` in this task.** Task 1.6 owns enforcement: `load_model_for(ticker)` returns `(model, feature_columns)` and its callers pass the computed frame through `align_to_schema`, failing the request on mismatch.

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_feature_schema.py -v`
Expected: PASS, all three.

- [ ] **Step 7: Commit**

```bash
git add src/features/schema.py src/train.py src/serve.py tests/test_feature_schema.py
git commit -m "feat: one explicit feature schema per model, enforced at load (F5)"
```

### Task 1.5: Remove standardisation from the final fit (F1)

**Files:**
- Modify: `src/train.py:361-365`
- Test: `tests/test_scale_invariance.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the final `XGBClassifier` (and the LightGBM blend) fitted on **raw** features; no `StandardScaler` in the final fit; no scaler persisted, because none is needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scale_invariance.py
import numpy as np, pandas as pd
from xgboost import XGBClassifier

def test_tree_predictions_invariant_to_monotone_rescale():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"vol": rng.lognormal(20, 1, 400), "rsi": rng.uniform(0, 100, 400)})
    y = (X["rsi"] > 50).astype(int)
    a = XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X, y)
    b = XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(X * 1000, y)
    assert np.array_equal(a.predict(X), b.predict(X * 1000))
```

- [ ] **Step 2: Run it and confirm it passes before the change**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_scale_invariance.py -v`
Expected: PASS. This test documents *why* the fix is to delete the scaler rather than persist it: the trained model is invariant to the rescale, so scaling only created a train/serve mismatch.

- [ ] **Step 3: Delete the scaler from the final fit**

In `src/train.py`, remove `sc_final = StandardScaler()`, `X_all_s = ...`, and fit on `X` directly. Update the LightGBM blend to fit and predict on `X`. Leave `prune_features`'s internal scaler alone — it is local to variance computation and never persisted.

- [ ] **Step 4: Run the tests and confirm the signal is no longer always SELL**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_scale_invariance.py tests/test_serve_parity.py -v`
Expected: PASS. After Task 2.3 retrains, `tests/test_not_all_sell.py` must show a non-zero BUY count on historical rows — that test is written in Task 2.2 and is the real proof.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_scale_invariance.py
git commit -m "fix(train): drop standardisation from the final fit (F1)

Trees are invariant to per-feature monotone rescaling, so the scaler bought
nothing and broke serving, which fed raw features to a model fitted on
z-scores. Removing it eliminates the mismatch rather than papering over it
with a new artefact to keep in sync."
```

### Task 1.6: One model selector that raises (F2, F9)

**Files:**
- Create: `src/model_store.py`
- Modify: `src/serve.py:216-267`, `src/backtest.py:88-116`, `cli.py:205-218`, `predict.py:101-117`
- Test: `tests/test_model_store.py`

**Interfaces:**
- Consumes: Task 1.4's `feature_schema.json`.
- Produces: `ModelNotFound(LookupError)`; `model_dir() -> Path`; `load_model_index() -> dict[str, str]`; `resolve_model_folder(ticker: str) -> str`; `load_model_for(ticker: str) -> tuple[Any, list[str]]` returning `(model, feature_columns)`, raising `ModelNotFound` when the index has no entry. **No fallback to any folder, ever.** All four call sites use this module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_store.py
import pytest
from model_store import ModelNotFound, resolve_model_folder

def test_unknown_ticker_raises(monkeypatch, tmp_path):
    import model_store
    monkeypatch.setattr(model_store, "load_model_index", lambda: {"BBCA.JK": "m-1"})
    with pytest.raises(ModelNotFound):
        resolve_model_folder("ZZZZ.JK")

def test_no_fallback_to_first_folder(monkeypatch):
    import model_store
    monkeypatch.setattr(model_store, "load_model_index", lambda: {})
    with pytest.raises(ModelNotFound):
        resolve_model_folder("BBCA.JK")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_model_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_store'`.

- [ ] **Step 3: Implement `src/model_store.py`**

Implement the five names above. The directory scan must exist only inside an explicit rebuild command, never inside a request path.

- [ ] **Step 4: Replace all four call sites**

In `src/serve.py` delete the `_fallback` and "scan all models" branches (lines 246-265). In `src/backtest.py` delete the `sorted(..., reverse=True)` fallback. In `cli.py` and `predict.py` route through `model_store`. The API and the backtester must now select identical models.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_model_store.py -v`
Expected: PASS, both tests.

- [ ] **Step 6: Commit**

```bash
git add src/model_store.py src/serve.py src/backtest.py cli.py predict.py tests/test_model_store.py
git commit -m "fix: single model selector that raises instead of substituting (F2, F9)"
```

### Task 1.7: Training writes the model index (F10)

**Files:**
- Modify: `src/train.py` (end of `main`), `scripts/build_model_index.py:69`
- Test: `tests/test_index_written.py`

**Interfaces:**
- Consumes: Task 1.6's `load_model_index`.
- Produces: `write_model_index(entries: dict[str, str], path: Path) -> None` invoked once at the end of training, so the index is authoritative and never empty after a successful retrain. `build_model_index.py` gains `mlflow.set_tracking_uri(...)` and, when a run cannot be resolved, logs at `warning` and **exits non-zero** instead of falling through to overlap guessing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_written.py
import json
from model_store import write_model_index, load_model_index

def test_index_roundtrip(tmp_path, monkeypatch):
    import model_store
    p = tmp_path / "model_index.json"
    monkeypatch.setattr(model_store, "index_path", lambda: p)
    write_model_index({"BBCA.JK": "m-1"}, p)
    assert json.loads(p.read_text())["ticker_to_model"] == {"BBCA.JK": "m-1"}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_index_written.py -v`
Expected: FAIL — `write_model_index` not defined.

- [ ] **Step 3: Implement and wire the single writer**

Implement `write_model_index`, preserving the existing JSON shape (`total_models`, `indexed_models`, `mapped_tickers`, `errors`, `ticker_to_model`) so the file format does not change. Call it at the end of `train.py:main` with the ticker→folder pairs it just produced. In `build_model_index.py`, set the tracking URI from `MLFLOW_DB` and replace the silent `except Exception: pass` around `client.get_run` with a `logger.warning` plus a non-zero exit when any model fails to resolve.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_index_written.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py src/model_store.py scripts/build_model_index.py tests/test_index_written.py
git commit -m "fix: training writes the model index; index rebuild fails loudly (F10)"
```

### Task 1.8: One cache path, aligned by name (F6, F7)

**Files:**
- Modify: `src/serve.py:154-186, 484-560`
- Test: `tests/test_prediction_cache.py`

**Interfaces:**
- Consumes: Task 1.4's `align_to_schema`.
- Produces: `_cache_put(ticker, feature_row: pd.Series, schema: list[str], window: str)` and `_cache_get(ticker, schema: list[str], window: str) -> pd.Series | None`, storing the feature values **keyed by schema name and fingerprinted by the schema itself and the window**. The fast and slow paths converge on one code path; a schema or window mismatch is a cache miss, never a positional zip.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prediction_cache.py
import pandas as pd

def test_fast_and_slow_path_agree(tmp_path, monkeypatch):
    import serve
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "c.db"))
    schema = ["volatility_20d", "rsi"]
    row = pd.Series({"volatility_20d": 9.0, "rsi": 0.1})
    serve._cache_put("BBCA.JK", row, schema, "0900")
    got = serve._cache_get("BBCA.JK", schema, "0900")
    assert got["volatility_20d"] == 9.0 and got["rsi"] == 0.1

def test_window_change_is_a_miss(tmp_path, monkeypatch):
    import serve
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "c.db"))
    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"], "0900")
    assert serve._cache_get("BBCA.JK", ["rsi"], "1330") is None

def test_schema_change_is_a_miss(tmp_path, monkeypatch):
    import serve
    monkeypatch.setattr(serve, "PRED_CACHE_DB", str(tmp_path / "c.db"))
    serve._cache_put("BBCA.JK", pd.Series({"rsi": 1.0}), ["rsi"], "0900")
    assert serve._cache_get("BBCA.JK", ["rsi", "volatility_20d"], "0900") is None
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_prediction_cache.py -v`
Expected: FAIL — the current schema is `(ticker, date, features BLOB, ohlcv BLOB)` with no window or schema fingerprint, and the test's new signature does not exist.

- [ ] **Step 3: Implement the new cache**

Add `CREATE TABLE IF NOT EXISTS cache (ticker TEXT, window TEXT, schema_hash TEXT, features TEXT, PRIMARY KEY (ticker, window, schema_hash))`. Store features as JSON keyed by column name, not `pickle` — a shared database must not contain deserialisable code. Delete the old table on first connect of the new version. Replace the positional zip at lines 493-497 with `align_to_schema`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_prediction_cache.py -v`
Expected: PASS, all three.

- [ ] **Step 5: Commit**

```bash
git add src/serve.py tests/test_prediction_cache.py
git commit -m "fix(serve): one cache path, name-aligned, no pickle (F6, F7)"
```

### Task 1.9: Remove dead calibration and make degradation visible (F8)

**Files:**
- Modify: `src/train.py:386-393`, `src/serve.py:427-452, 650-659`, `README.md:305`
- Test: `tests/test_degradation_visible.py`

**Interfaces:**
- Consumes: nothing.
- Produces: no `CalibratedCVClassifier` reference anywhere; `PredictResponse` gains `degraded: list[str]` naming every component that failed (e.g. `["sentiment"]`); the batch endpoint raises HTTP 502 for a failed ticker rather than returning `prediction: 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_degradation_visible.py
import pytest
from fastapi.testclient import TestClient

def test_sentiment_failure_is_flagged(monkeypatch):
    import serve
    monkeypatch.setattr(serve, "build_inference_features",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("news api down")))
    client = TestClient(serve.app)
    r = client.post("/predict", json={"ticker": "BBCA.JK"})
    assert r.status_code in (500, 502)
    assert "sentiment" in r.text or "news api down" in r.text

def test_no_dead_calibration_symbol():
    import pathlib
    src = pathlib.Path("src/train.py").read_text()
    assert "CalibratedCVClassifier" not in src
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_degradation_visible.py -v`
Expected: FAIL — `CalibratedCVClassifier` is present at `train.py:388`, and a downstream feature failure currently surfaces as a generic 500 without naming sentiment.

- [ ] **Step 3: Delete the calibration block and fix the README**

Remove lines 386-393 entirely. In `README.md`, delete the isotonic-calibration claim and the `calibrated` row. If calibration is wanted later, it is a separate, tested change — it has never run, so removing it makes the docs true.

- [ ] **Step 4: Make degradation explicit**

Add `degraded: list[str] = []` to `PredictResponse`. In the sentiment overlay, append `"sentiment"` and log at `warning` when the fetch fails. Change the batch failure payload to `raise HTTPException(status_code=502, detail=...)` instead of returning a signal-shaped object.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_degradation_visible.py -v`
Expected: PASS, both tests.

- [ ] **Step 6: Commit**

```bash
git add src/train.py src/serve.py README.md tests/test_degradation_visible.py
git commit -m "fix: remove dead calibration, surface degradation, batch errors are errors (F8)"
```

### Task 1.10: Exchange calendar and WIB clock

**Files:**
- Create: `src/market_calendar.py`
- Modify: `src/serve.py` (response), `requirements.txt`
- Test: `tests/test_market_calendar.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `is_trading_day(d: date) -> bool`; `signal_windows(d: date) -> list[tuple[str, datetime]]` returning the three WIB instants for that date; `SessionWindow` constants. Mon–Thu session 1 is 09:00–12:00 and session 2 is 13:30–15:49:59; **Friday session 1 is 09:00–11:30 and session 2 is 14:00–15:49:59**. Window keys: `"open"` = open + 10 min, `"post_lunch"` = session 2 start + 10 min, `"pre_close"` = 15:40.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_market_calendar.py
from datetime import date
from market_calendar import signal_windows, is_trading_day

def test_monday_thursday_lunch_window():
    w = dict(signal_windows(date(2026, 10, 13)))  # a Tuesday
    assert w["open"].strftime("%H:%M") == "09:10"
    assert w["post_lunch"].strftime("%H:%M") == "13:40"
    assert w["pre_close"].strftime("%H:%M") == "15:40"

def test_friday_lunch_window_differs():
    w = dict(signal_windows(date(2026, 10, 16)))  # a Friday
    assert w["post_lunch"].strftime("%H:%M") == "14:10"

def test_weekend_is_not_a_trading_day():
    assert not is_trading_day(date(2026, 10, 17))
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_market_calendar.py -v`
Expected: FAIL — no such module. The repo currently has zero timezone or holiday handling anywhere.

- [ ] **Step 3: Implement the calendar**

Implement using `zoneinfo.ZoneInfo("Asia/Jakarta")` and an IDX holiday list. Add the exchange-calendar dependency you choose to `requirements.txt` with a `==` pin. Holidays must be data, not guesses — list them explicitly with a comment naming the source, and make `is_trading_day` return `False` for any date not positively known to be a trading day.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_market_calendar.py -v`
Expected: PASS, all three.

- [ ] **Step 5: Commit**

```bash
git add src/market_calendar.py tests/test_market_calendar.py requirements.txt
git commit -m "feat: IDX trading calendar and WIB signal windows"
```

---

## Phase 2 — Prove it

### Task 2.1: End-to-end API test

**Files:**
- Test: `tests/test_api_end_to_end.py`

**Interfaces:**
- Consumes: everything in Phase 1.
- Produces: a test that boots the real FastAPI app with `TestClient`, retrains nothing, and asserts the full contract on a real model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_end_to_end.py
from fastapi.testclient import TestClient
import serve

def test_predict_contract():
    client = TestClient(serve.app)
    r = client.post("/predict", json={"ticker": "BBCA.JK"})
    assert r.status_code == 200, r.text
    body = r.json()
    for field in ["ticker", "prediction", "probability_up", "signal",
                  "sentiment_score", "probability_adjusted", "signal_adjusted", "degraded"]:
        assert field in body, field
    assert body["signal"] in ("BUY", "SELL")
    assert 0.0 <= body["probability_up"] <= 1.0

def test_unknown_ticker_is_an_error_not_a_guess():
    client = TestClient(serve.app)
    r = client.post("/predict", json={"ticker": "ZZZZ.JK"})
    assert r.status_code in (404, 422)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_api_end_to_end.py -v`
Expected: FAIL until Task 2.3 retrains and rebuilds the index — before that, `BBCA.JK` has no index entry and the request must 404, which is itself the correct behaviour.

- [ ] **Step 3: Commit once green after Task 2.3**

```bash
git add tests/test_api_end_to_end.py
git commit -m "test: end-to-end API contract"
```

### Task 2.2: Regression suite for the audit's critical findings

**Files:**
- Test: `tests/test_audit_regressions.py`

**Interfaces:**
- Consumes: all of Phase 1.
- Produces: one test per critical/high finding, each named for the finding, so the audit's conclusions become executable.

- [ ] **Step 1: Write one test per finding**

```python
# tests/test_audit_regressions.py
# F1: model no longer always says SELL on historical data
def test_not_all_sell_on_history(): ...

# F2: two different tickers must not receive the same model
def test_distinct_tickers_use_distinct_models(): ...

# F3/F5: no feature the model expects is all-NaN at inference
def test_no_all_nan_model_feature(): ...

# F6/F7: cache round-trip preserves per-name values
def test_cache_by_name(): ...

# F8: calibration symbol gone, degradation flagged
def test_degraded_flag_present(): ...

# F11: no undeclared import
def test_all_imports_declared(): ...

# F12: batch failure is not a signal
def test_batch_failure_is_error(): ...
```

- [ ] **Step 2: Run the suite**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_audit_regressions.py -v`
Expected: PASS. Any failure here is a finding that survived the fix — do not adjust the assertion to make it green.

- [ ] **Step 3: Commit**

```bash
git add tests/test_audit_regressions.py
git commit -m "test: regression suite pinning every audit critical and high finding"
```

### Task 2.3: Retrain, rebuild the index, verify the mapping

**Files:**
- No source changes; produces `mlruns/1/models/**` and `mlruns/1/models/model_index.json`

**Interfaces:**
- Consumes: the corrected training path.
- Produces: 45 models trained on raw features with `feature_schema.json` beside each, and an index mapping all 45 tickers.

- [ ] **Step 1: Quarantine the old models**

Move `mlruns/1/models` to `mlruns/1/models.broken-2026-10-13`. They were fitted on a lost transform; keeping them loadable invites exactly the silent substitution this plan removes.

- [ ] **Step 2: Retrain**

Run: `.\venv\Scripts\python.exe src/train.py --parallel --features data/processed/features.parquet`
Expected: `Done: 45/45 tickers trained`. The parquet is the correct input; the CSV path is the stale-data trap.

- [ ] **Step 3: Verify the index is populated and correct**

Run: `.\venv\Scripts\python.exe -c "import json; d=json.load(open('mlruns/1/models/model_index.json')); print(d['mapped_tickers'], len(d['ticker_to_model']))"`
Expected: `45 45`

- [ ] **Step 4: Spot-check the mapping is real, not guessed**

For three tickers, confirm the model's own `feature_schema.json` matches the columns computed for that ticker, and that two different tickers resolve to different folders:
Run: `.\venv\Scripts\python.exe -m pytest tests/test_audit_regressions.py::test_distinct_tickers_use_distinct_models -v`
Expected: PASS.

- [ ] **Step 5: Run the whole suite**

Run: `.\venv\Scripts\python.exe -m pytest -v`
Expected: all tests pass, including `tests/test_api_end_to_end.py`.

- [ ] **Step 6: Commit**

```bash
git add mlruns/1/models/model_index.json
git commit -m "chore: retrain 45 models on raw features with enforced schemas"
```

### Task 2.4: Make a fundamentals outage visible (added during execution)

**Files:**
- Modify: `src/features/fetchers.py`, `src/features/pipeline.py`, `src/serve.py`
- Test: `tests/test_fundamentals_visible.py`

**Interfaces:**
- Consumes: Task 1.9's `degraded` mechanism and its `fetch_news_sentiment_status` precedent.
- Produces: `fetch_fundamentals_status(ticker) -> (dict, bool, str | None)` mirroring the sentiment seam; `PredictResponse.degraded` naming `"fundamentals"` when the source failed; `dividend_yield` no longer fabricated.

**Why this task exists.** Task 2.3's retrain made the 60-name feature contract unconditional for all 45 models, which turned two latent gaps into certain ones. Both are the same "a safe default indistinguishable from a real value" family as F8:

1. `src/features/fetchers.py` returns `{key: np.nan}` for all eight fundamental keys when `yf.Ticker(t).info` raises. `pipeline.inject_macro_features` materialises the five schema-required fundamentals as `NaN`, and `align_to_schema` checks **presence only** — so a Yahoo `.info` outage returns HTTP 200 with a real probability computed on five missing inputs and `degraded: []`. `usdidr_return` is a third instance and *is* in the schema.
2. `src/features/fetchers.py:48` reads `info.get("dividendYield", np.nan) or 0.0`. `np.nan` is truthy, so the fallback fires on `None` — yfinance's "unknown" — turning an unknown into a fabricated `0.0` in a consumed column. XGBoost reads `0.0` and `NaN` differently.

- [ ] **Step 1: Write the failing test**

Drive the **real** fetcher with a `yf.Ticker(...).info` that raises, then assert the response carries `degraded` containing `"fundamentals"` and a `warning` naming the reason; separately assert an explicit `None` dividend yield reaches the frame as `NaN`, not `0.0`.

- [ ] **Step 2: Confirm RED**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_fundamentals_visible.py -v`
Expected: FAIL — no status seam exists and `dividend_yield` reads `0.0`.

- [ ] **Step 3: Implement the seam and drop the fabrication**

Add `fetch_fundamentals_status` returning `(values, available, reason)` in the sentiment seam's shape, so callers read a status rather than catching an exception. Keep `fetch_fundamentals` working for existing callers. Drop the `or 0.0`. Thread availability into `degraded` from the serving path.

- [ ] **Step 4: Confirm GREEN and that the success path is unchanged**

Run the new test, then the full suite. A healthy `.info` must still yield the same eight values and `degraded == []`.

- [ ] **Step 5: Commit**

```bash
git add src/features/fetchers.py src/features/pipeline.py src/serve.py tests/test_fundamentals_visible.py
git commit -m "fix: make a fundamentals outage visible and stop fabricating dividend yield"
```

---

## Phase 3 — Re-measure (decision gate)

### Task 3.1: Realistic IDX costs in the backtest

**Files:**
- Modify: `src/backtest.py:52, 205-260`
- Test: `tests/test_backtest_costs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DEFAULT_BUY_COST = 0.0015`, `DEFAULT_SELL_COST = 0.0025` (brokerage plus the 0.1% final sales tax), and `slippage_bps: int = 0` applied per side. The existing `DEFAULT_COST_PER_TRADE` is removed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_costs.py
from backtest import costs_for

def test_sell_costs_more_than_buy():
    assert costs_for(1_000_000, side="sell") > costs_for(1_000_000, side="buy")

def test_slippage_is_applied_per_side():
    assert costs_for(1_000_000, side="buy", slippage_bps=10) > costs_for(1_000_000, side="buy")
```

- [ ] **Step 2: Run it, confirm it fails, implement, confirm it passes**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_backtest_costs.py -v` (FAIL), implement `costs_for`, re-run (PASS).

- [ ] **Step 3: Commit**

```bash
git add src/backtest.py tests/test_backtest_costs.py
git commit -m "fix(backtest): model the IDX sell tax and slippage, not one flat fee"
```

### Task 3.2: The honest measurement

**Files:**
- Create: `docs/measurement-2026-10-13.md`

**Interfaces:**
- Consumes: Tasks 2.3 and 3.1.
- Produces: a written result: per-ticker and aggregate return, Sharpe, max drawdown, trade count, and total costs paid, at `slippage_bps` of 0, 10 and 25.

- [ ] **Step 1: Run the backtest at three cost levels**

Run: `.\venv\Scripts\python.exe src/backtest.py --threshold 0.50 --slippage-bps 0`, then `10`, then `25`.
Expected: three result sets. Record them verbatim.

- [ ] **Step 2: Write the report**

State plainly whether the strategy survives costs, and at what turnover. Do not compute a headline number without the cost column beside it.

- [ ] **Step 3: Commit**

```bash
git add docs/measurement-2026-10-13.md
git commit -m "docs: honest post-fix measurement at three cost levels"
```

### Task 3.3: Decision gate (no code)

- [ ] **Step 1: Answer three questions in writing**

1. Does the signal survive realistic IDX costs? If not, the model — not the dashboard — is the next project.
2. Is the daily (next-day close) target the right one, or does it need to become intraday? The current answer is that three runs a day return the same value, because the target is daily and the cache is daily.
3. Are the signals worth a dashboard yet?

- [ ] **Step 2: Only on a yes to (3), plan Phase 4**

Phase 4 — the web dashboard replacing Grafana, the free database plus Drive cold storage, the dynamic LQ45 universe, and the Clef confidence layer — is a **separate plan** and must be written after this gate, because its shape depends on the answers. It is deliberately not specified here.

---

## Self-Review

**Spec coverage.** F1 → 1.5; F2 → 1.6; F3 → 1.3; F4 → 0.1, 0.2; F5 → 1.3, 1.4; F6 → 1.8; F7 → 1.8; F8 → 1.9; F9 → 1.6; F10 → 1.7; F11 → 0.3; F12 → 1.9. Medium findings: dependency scanning → 0.3; mutable images → 0.3; dev tools → 1.1; malformed `.gitignore` → 0.4; sentiment invisibility → 1.9; README drift → 0.4, 1.9; `ADMR.JK` sector → 0.4; `mlruns/1` hardcoding → **not fixed by design** (constraint: location unchanged; it is a Known Issue for Phase 4); intraday leakage → **deferred to the Phase 3 gate**, because it only matters if the target becomes intraday; Docker `:latest` → 0.3. Low findings: Grafana admin password → superseded by Phase 4; host username in `mlflow.db` → resolved by 0.2 (stop committing it); unused deps → 0.3; `--tune-all` and `--parallel` flag handling → 0.4 documents it, fixing it is Phase 4 scope; global warning suppression → 1.1 (`filterwarnings = ["error"]`).

**Gaps accepted deliberately.** No task makes `start.bat` option 7/8 equivalent to the README path beyond reachability; no task adds the news-sentiment vendor dependency to the venv (Task 0.3 declares it, installing it is Task 1.1's `requirements-dev.txt` step plus a `requirements.txt` install). No task addresses `usdidr_rate` being cache-dependent — Task 1.3's staleness signal is the mitigation.

**Type consistency.** `build_features` (Task 1.2) is consumed by 1.3 and 2.2 with identical keywords. `align_to_schema`/`FeatureSchemaError` (1.4) are consumed by 1.8. `ModelNotFound`/`resolve_model_folder`/`load_model_for` (1.6) are consumed by 1.7, 2.2 and all four replaced call sites. `write_model_index` (1.7) is distinct from `load_model_index` (1.6) — both live in `model_store.py`. `costs_for` (3.1) is new and used only there.

**Review Focus coverage.** 1 → 1.6; 2 → 1.4; 3 → 1.3; 4 → 1.8; 5 → 1.10; 6 → 1.9.

**Proportion.** Every step is a signature, a test, or a command. Function bodies appear only where an algorithm is not determined by the signature and tests (`align_to_schema`, the calendar).

## Known issues left open

- `mlruns/1` remains a hardcoded experiment id in 12 places (constraint: unchanged this plan).
- Per-ticker models are retained; a pooled model with ticker embeddings may supersede them in Phase 4 if the measurement is weak.
- `dvc` is removed from the manifest; the tracked `.dvc` files and the unreachable remote remain until Phase 4 decides DVC's fate.
- The prediction cache is keyed by window, but nothing yet *schedules* the three windows — scheduling belongs to Phase 4, since it depends on where the signal job runs (GitHub Actions cron is not a reliable clock for a 09:10 promise).
