import sqlite3

db_path = '/app/mlflow.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check tags for run_uuid mapping
c.execute("SELECT key, value, run_uuid FROM tags LIMIT 30")
print("Tags (first 30):")
for row in c.fetchall():
    print(f"  {row[0]} = {row[1]} (run={row[2][:8]}...)")

# Check params
c.execute("SELECT key, value, run_uuid FROM params LIMIT 20")
print("\nParams (first 20):")
for row in c.fetchall():
    print(f"  {row[0]} = {row[1]} (run={row[2][:8]}...)")

# Check logged_models table
c.execute("SELECT * FROM logged_models LIMIT 5")
cols = [d[0] for d in c.description]
print(f"\nLogged models cols: {cols}")
for row in c.fetchall():
    print(f"  {row}")

# Count runs per experiment
c.execute("SELECT experiment_id, COUNT(*) FROM runs GROUP BY experiment_id")
print(f"\nRuns per experiment: {c.fetchall()}")

# Get experiment names
c.execute("SELECT experiment_id, name FROM experiments")
print(f"Experiments: {c.fetchall()}")

conn.close()
