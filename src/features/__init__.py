"""
Features package — modular feature engineering for Indonesian Stock MLOps.

Modules:
    fetchers       — fetch_fundamentals, fetch_usdidr, fetch_fred_macro, etc.
    indicators     — compute_ta_features, compute_custom_features
    enhanced_mas   — compute_enhanced_mas
    ict            — compute_ict_features (ICT Smart Money Concepts)
    volume_profile — compute_volume_profile_features
    market         — load_ihsg_data, compute_market_context, compute_cross_stock_features
    pipeline       — load_data, inject_macro_features, compute_target,
                     engineer_features_for_ticker, build_feature_set

Usage (same as before):
    from features import compute_ta_features, fetch_fundamentals, ...
"""
# sys.path setup so submodules can import from src/
import sys as _sys
import os as _os
_curr = _os.path.dirname(_os.path.abspath(__file__))
_parent = _os.path.dirname(_curr)
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

# ── Re-export data fetchers ───────────────────────────────────────────────────
from .fetchers import (
    fetch_fundamentals,
    fetch_usdidr,
    fetch_fred_macro,
    fetch_bi_rate,
    fetch_google_trends,
    fetch_news_sentiment,
    fetch_idx_fundamentals,
    FRED_AVAILABLE,
    TRENDS_AVAILABLE,
)

# ── Re-export TA indicators ───────────────────────────────────────────────────
from .indicators import (
    compute_ta_features,
    compute_custom_features,
)

# ── Re-export enhanced MAs ────────────────────────────────────────────────────
from .enhanced_mas import (
    compute_enhanced_mas,
)

# ── Re-export ICT features ────────────────────────────────────────────────────
from .ict import (
    compute_ict_features,
)

# ── Re-export volume profile ──────────────────────────────────────────────────
from .volume_profile import (
    compute_volume_profile_features,
)

# ── Re-export market context ──────────────────────────────────────────────────
from .market import (
    load_ihsg_data,
    compute_market_context,
    compute_cross_stock_features,
)

# ── Re-export pipeline ────────────────────────────────────────────────────────
from .pipeline import (
    load_data,
    inject_macro_features,
    compute_target,
    engineer_features_for_ticker,
    build_feature_set,
)

__all__ = [
    "fetch_fundamentals", "fetch_usdidr", "fetch_fred_macro", "fetch_bi_rate",
    "fetch_google_trends", "fetch_news_sentiment", "fetch_idx_fundamentals",
    "compute_ta_features", "compute_custom_features",
    "compute_enhanced_mas", "compute_ict_features",
    "compute_volume_profile_features",
    "load_ihsg_data", "compute_market_context", "compute_cross_stock_features",
    "load_data", "inject_macro_features", "compute_target",
    "engineer_features_for_ticker", "build_feature_set",
]
