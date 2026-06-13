"""
Convenience script for running Optuna hyperparameter tuning.

Usage:
    python scripts/tune.py                 # Tune top-10 liquid tickers
    python scripts/tune.py --all           # Tune ALL 45 tickers (takes hours!)
    python scripts/tune.py --ticker BBCA.JK
    python scripts/tune.py --trials 50     # More trials per ticker
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning")
    parser.add_argument('--ticker', type=str, help='Single ticker to tune')
    parser.add_argument('--all', action='store_true', help='Tune ALL 45 tickers')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials per ticker (default: 20)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would run')
    args = parser.parse_args()

    train_script = os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py')

    if args.ticker:
        cmd = [sys.executable, train_script, '--tune', '--ticker', args.ticker]
        print(f"Tuning single ticker: {args.ticker} ({args.trials} trials)")
        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            subprocess.run(cmd)

    elif args.all:
        cmd = [sys.executable, train_script, '--tune']
        print(f"Tuning ALL 45 tickers ({args.trials} trials each)")
        print("WARNING: This will take 2-6 hours on CPU!")
        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            subprocess.run(cmd)

    else:
        # Default: tune top-10 (train.py limits to TRAINING_CONFIG['tune_top_n_tickers'])
        cmd = [sys.executable, train_script, '--tune']
        print(f"Tuning top-10 most liquid tickers ({args.trials} trials each)")
        print("Estimated time: ~30-60 minutes")
        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
        else:
            subprocess.run(cmd)


if __name__ == '__main__':
    main()
