import sqlite3
from collections import defaultdict

import os
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlflow.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Get backtest runs - join params and metrics
c.execute("""
    SELECT p.value, m.key, m.value
    FROM runs r
    JOIN params p ON r.run_uuid = p.run_uuid AND p.key = 'ticker'
    JOIN metrics m ON r.run_uuid = m.run_uuid
    WHERE r.experiment_id = (SELECT experiment_id FROM experiments WHERE name='backtest')
    ORDER BY r.start_time
""")
rows = c.fetchall()
print(f'Total metric rows: {len(rows)}')

# Group by ticker
results = defaultdict(dict)
for ticker, metric_key, metric_val in rows:
    results[ticker][metric_key] = metric_val

print(f'\n=== Backtest Results (threshold=0.50) ===')
print(f'{"Ticker":10s} {"Return":>8s} {"Sharpe":>8s} {"MaxDD":>8s} {"WinRate":>8s} {"Trades":>7s}')
print('-' * 55)
for ticker in sorted(results.keys()):
    m = results[ticker]
    r = float(m.get('total_return', 0)) * 100
    s = float(m.get('sharpe_ratio', 0))
    d = float(m.get('max_drawdown', 0)) * 100
    w = float(m.get('win_rate', 0)) * 100
    n = int(m.get('n_trades', 0))
    print(f'{ticker:10s} {r:>7.1f}% {s:>8.2f} {d:>7.1f}% {w:>7.0f}% {n:>7d}')

# Summary stats
returns = [float(results[t].get('total_return', 0)) for t in results]
sharpe = [float(results[t].get('sharpe_ratio', 0)) for t in results]
print(f'\n=== Summary ===')
print(f'Tickers tested: {len(results)}')
print(f'Mean return: {sum(returns)/len(returns)*100:.2f}%')
median_idx = len(returns) // 2
sorted_returns = sorted(returns)
print(f'Median return: {sorted_returns[median_idx]*100:.2f}%')
print(f'Best return: {max(returns)*100:.2f}%')
print(f'Worst return: {min(returns)*100:.2f}%')
print(f'Positive tickers: {sum(1 for r in returns if r > 0)}/{len(returns)}')
print(f'Mean Sharpe: {sum(sharpe)/len(sharpe):.2f}')
print(f'Max Sharpe: {max(sharpe):.2f}')
print(f'Min Sharpe: {min(sharpe):.2f}')

# Top and bottom 5
print(f'\n=== Top 5 by Return ===')
top5 = sorted(results.items(), key=lambda x: float(x[1].get('total_return', 0)), reverse=True)[:5]
for ticker, m in top5:
    r = float(m.get('total_return', 0)) * 100
    s = float(m.get('sharpe_ratio', 0))
    print(f'  {ticker:10s} {r:>7.1f}% Sharpe: {s:.2f}')

print(f'\n=== Bottom 5 by Return ===')
bot5 = sorted(results.items(), key=lambda x: float(x[1].get('total_return', 0)))[:5]
for ticker, m in bot5:
    r = float(m.get('total_return', 0)) * 100
    s = float(m.get('sharpe_ratio', 0))
    print(f'  {ticker:10s} {r:>7.1f}% Sharpe: {s:.2f}')

conn.close()
