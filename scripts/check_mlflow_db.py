import sqlite3, os

db_path = '/app/mlflow.db'
print(f'mlflow.db exists: {os.path.exists(db_path)}')
print(f'models count: {len(os.listdir("/app/mlruns/1/models/"))}')

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Show tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(f'Tables: {tables}')

# Show columns of 'runs' table
if 'runs' in tables:
    c.execute("PRAGMA table_info(runs)")
    cols = [(r[1], r[2]) for r in c.fetchall()]
    print(f'runs columns: {[c[0] for c in cols]}')

# Try a query
if 'runs' in tables:
    try:
        c.execute("SELECT run_uuid, experiment_id FROM runs ORDER BY start_time DESC LIMIT 5")
        print(f'Runs: {c.fetchall()}')
    except:
        c.execute("SELECT * FROM runs LIMIT 5")
        row = c.fetchone()
        if row:
            print(f'First run col names: {[d[0] for d in c.description]}')
            print(f'First run: {row}')

# Check experiments table
if 'experiments' in tables:
    c.execute("SELECT * FROM experiments")
    print(f'Experiments: {c.fetchall()}')

conn.close()
