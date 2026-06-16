"""CI entry point for feature engineering."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.pipeline import build_feature_set
from config import TICKERS, DATA_PROCESSED_CSV_PATH
build_feature_set('data/raw/stocks.csv', DATA_PROCESSED_CSV_PATH, TICKERS, mode='ci')
