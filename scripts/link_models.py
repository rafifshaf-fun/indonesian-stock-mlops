"""
Create human-readable model directories under models/by_ticker/

Creates symlinks (or copies on Windows) from the UUID-named MLflow
model folders to ticker-named directories.

Usage:
    python scripts/link_models.py
"""
import sys, os, json, shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'mlruns', '1', 'models')
INDEX_FILE = os.path.join(MODELS_DIR, 'model_index.json')
BY_TICKER_DIR = os.path.join(BASE_DIR, 'models', 'by_ticker')


def link_models():
    if not os.path.exists(INDEX_FILE):
        print(f"Error: {INDEX_FILE} not found. Run build_model_index.py first.")
        sys.exit(1)

    with open(INDEX_FILE) as f:
        data = json.load(f)

    ticker_to_model = data.get('ticker_to_model', {})
    os.makedirs(BY_TICKER_DIR, exist_ok=True)

    created = 0
    for ticker, info in ticker_to_model.items():
        src_folder = info['model_folder']
        src_path = os.path.join(MODELS_DIR, src_folder)
        dst_path = os.path.join(BY_TICKER_DIR, ticker)

        if not os.path.isdir(src_path):
            print(f"  SKIP {ticker}: source folder {src_folder} not found")
            continue

        # Remove existing if stale
        if os.path.exists(dst_path):
            if os.path.islink(dst_path) or os.path.isdir(dst_path):
                if os.path.islink(dst_path):
                    os.unlink(dst_path)
                else:
                    shutil.rmtree(dst_path)

        # Try symlink first, fall back to copy on Windows
        try:
            os.symlink(src_path, dst_path, target_is_directory=True)
            print(f"  SYMLINK {ticker} -> {src_folder}")
        except (OSError, NotImplementedError):
            # Windows without admin: copy the artifacts directory
            src_artifacts = os.path.join(src_path, 'artifacts')
            dst_artifacts = os.path.join(dst_path, 'artifacts')
            os.makedirs(dst_path, exist_ok=True)
            if os.path.isdir(dst_artifacts):
                shutil.copytree(src_artifacts, dst_artifacts, dirs_exist_ok=True)
            # Also copy MLmodel metadata
            for fname in ['MLmodel', 'conda.yaml', 'python_env.yaml', 'requirements.txt']:
                src_file = os.path.join(src_artifacts, fname)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(dst_artifacts, fname))
            print(f"  COPY {ticker} <- {src_folder}")
        created += 1

    print(f"\nDone: {created} tickers linked in {BY_TICKER_DIR}")
    print(f"Use: models/by_ticker/BBCA.JK/")


if __name__ == '__main__':
    link_models()
