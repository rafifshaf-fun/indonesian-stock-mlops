"""
Central configuration for the Indonesian Stock MLOps Pipeline.
Single source of truth for tickers, sectors, paths, and feature flags.
"""

# ── LQ45 Tickers (as of February 2026) ────────────────────────────────────────
TICKERS = [
    "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK",
    "AMRT.JK", "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK",
    "BBNI.JK", "BBRI.JK", "BBTN.JK", "BMRI.JK", "BREN.JK",
    "BRIS.JK", "BRPT.JK", "CPIN.JK", "CTRA.JK", "EXCL.JK",
    "GOTO.JK", "ICBP.JK", "INCO.JK", "INDF.JK", "INKP.JK",
    "ISAT.JK", "ITMG.JK", "JPFA.JK", "JSMR.JK", "KLBF.JK",
    "MAPA.JK", "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK",
    "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK",
    "SMRA.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK",
]

# ── Sector Groupings (for cross-stock features) ───────────────────────────────
SECTORS = {
    "banking": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BBTN.JK", "BRIS.JK", "ARTO.JK"],
    "energy_mining": ["ADRO.JK", "ANTM.JK", "PTBA.JK", "ITMG.JK", "MEDC.JK", "PGEO.JK", "INCO.JK", "MDKA.JK", "MBMA.JK", "BREN.JK", "AADI.JK", "AKRA.JK"],
    "telco": ["TLKM.JK", "ISAT.JK", "EXCL.JK", "TOWR.JK"],
    "consumer": ["UNVR.JK", "ICBP.JK", "INDF.JK", "JPFA.JK", "CPIN.JK", "KLBF.JK", "SIDO.JK", "AMRT.JK", "MAPI.JK", "MAPA.JK"],
    "infra_construction": ["SMGR.JK", "CTRA.JK", "SMRA.JK", "JSMR.JK", "PGAS.JK"],
    "conglomerate_auto": ["ASII.JK", "UNTR.JK", "INKP.JK", "BRPT.JK"],
    "tech_other": ["GOTO.JK", "AMMN.JK"],
}

# ── Market Index ───────────────────────────────────────────────────────────────
IHSG_TICKER = "^JKSE"

# ── Date & Path Configuration ──────────────────────────────────────────────────
START_DATE = "2020-01-01"
DATA_RAW_PATH = "data/raw/stocks.csv"
DATA_PROCESSED_PATH = "data/processed/features.parquet"
DATA_PROCESSED_CSV_PATH = "data/processed/features.csv"  # legacy CSV fallback
INTRADAY_CACHE_DIR = "data/intraday"
MODEL_DIR = "mlruns"
MLFLOW_EXPERIMENT = "indonesian-stock-prediction"
MLFLOW_DB = "mlflow.db"

# ── FRED Macro Series ──────────────────────────────────────────────────────────
FRED_SERIES = {
    "wti_oil": "DCOILWTICO",
    "gold_price": "GOLDAMGBD228NLBM",
    "coal_price": "PCOALAUUSDM",
    "nickel_price": "PNICKUSDM",
    "fed_rate": "FEDFUNDS",
    "vix": "VIXCLS",
    "us10y": "DGS10",
    "bi_rate_proxy": "IR3TIB01IDM156N",
}

# ── Google Trends Search Terms ─────────────────────────────────────────────────
TICKER_SEARCH_TERMS = {
    "BBCA.JK": "Bank BCA", "BBRI.JK": "Bank BRI", "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank BNI", "BBTN.JK": "Bank BTN", "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International", "GOTO.JK": "GoTo Gojek", "BREN.JK": "Barito Renewables",
    "ADRO.JK": "Adaro Energy", "ANTM.JK": "Aneka Tambang", "PTBA.JK": "Bukit Asam",
    "INDF.JK": "Indofood", "ICBP.JK": "Indofood CBP", "KLBF.JK": "Kalbe Farma",
    "UNVR.JK": "Unilever Indonesia", "UNTR.JK": "United Tractors",
    "AMMN.JK": "Amman Mineral", "INKP.JK": "Indah Kiat Pulp", "ISAT.JK": "Indosat Ooredoo",
    "EXCL.JK": "XL Axiata", "TOWR.JK": "Sarana Menara", "PGAS.JK": "Perusahaan Gas Negara",
    "JSMR.JK": "Jasa Marga", "CPIN.JK": "Charoen Pokphand", "JPFA.JK": "JAPFA Comfeed",
    "SMGR.JK": "Semen Indonesia", "CTRA.JK": "Ciputra Development",
    "SMRA.JK": "Summarecon Agung", "BRIS.JK": "Bank Syariah Indonesia",
    "ARTO.JK": "Bank Jago", "PGEO.JK": "Pertamina Geothermal",
    "AADI.JK": "Adaro Andalan", "ADMR.JK": "Adaro Minerals", "AKRA.JK": "AKR Corporindo",
    "AMRT.JK": "Sumber Alfaria Trijaya", "BRPT.JK": "Barito Pacific",
    "MAPA.JK": "MAP Aktif Adiperkasa", "MAPI.JK": "Mitra Adiperkasa",
    "MBMA.JK": "Merdeka Battery", "MDKA.JK": "Merdeka Copper",
    "MEDC.JK": "Medco Energi", "PGEO.JK": "Pertamina Geothermal",
}

# ── Feature Flags (toggle feature categories on/off) ───────────────────────────
FEATURE_FLAGS = {
    "ta_indicators": True,          # ~75 TA indicators from ta library
    "enhanced_mas": True,           # 15 enhanced moving average features
    "volume_profile": True,         # 20 intraday volume profile features
    "ict_suite": True,              # 25 ICT (Smart Money Concepts) features
    "market_context": True,         # 15 market context + cross-stock features
    "idx_fundamentals": True,       # 10 supplementary IDX fundamentals
    "google_trends": True,          # Google Trends sentiment
    "fred_macro": True,             # FRED macroeconomic indicators
    "bi_rate": True,                # Bank Indonesia rate scraping
    "news_sentiment": True,          # NewsAPI + VADER sentiment scores
}

# ── Cache Configuration ────────────────────────────────────────────────────────
CACHE_CONFIG = {
    "intraday_ttl_hours": 4,        # Refetch intraday data if older than 4h
    "fundamentals_ttl_hours": 24,   # Refetch fundamentals daily
    "macro_ttl_hours": 6,           # Refetch macro data every 6 hours
    "rate_limit_delay_seconds": 2,  # Delay between Yahoo Finance API calls
    "max_batch_size": 10,           # Max tickers per batch fetch
}

# ── Training Configuration ─────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "min_rows": 100,                # Minimum rows per ticker to train
    "cv_splits": 5,                 # TimeSeriesSplit folds
    "cv_gap_days": 10,              # Embargo gap between train/val
    "xgb_params": {                 # Base XGBoost hyperparameters
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    },
    "corr_threshold": 0.95,         # Feature correlation pruning threshold
    "var_threshold": 0.001,         # Low-variance feature removal threshold
    "optuna_trials": 20,            # Hyperparameter tuning trials (--tune flag)
    "tune_top_n_tickers": 10,       # Only tune top-N most liquid tickers
}

# ── Logging Configuration ──────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)