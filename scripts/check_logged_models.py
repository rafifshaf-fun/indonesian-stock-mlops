import sqlite3, os, json

db_path = os.path.join(os.path.dirname(__file__), '..', 'mlflow.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check logged_models table
c.execute("PRAGMA table_info(logged_models)")
cols = [(r[1], r[2]) for r in c.fetchall()]
print(f"Logged models columns: {[c[0] for c in cols]}")

# Get all logged models
c.execute("SELECT * FROM logged_models")
rows = c.fetchall()
print(f"\nLogged models count: {len(rows)}")
for row in rows:
    print(f"  model_id={row[0]}, experiment_id={row[1]}, name={row[2]}, source_run_id={row[9]}")

# Check model_version_tags
c.execute("PRAGMA table_info(model_version_tags)")
cols2 = [(r[1], r[2]) for r in c.fetchall()]
print(f"\nmodel_version_tags columns: {[c[0] for c in cols2]}")
c.execute("SELECT * FROM model_version_tags")
for row in c.fetchall():
    print(f"  {row}")

# Check registered_models
c.execute("SELECT * FROM registered_models")
for row in c.fetchall():
    print(f"\nRegistered model: {row}")

conn.close()

# Also check MLmodel files for run_id
models_dir = os.path.join(os.path.dirname(__file__), '..', 'mlruns', '1', 'models')
if os.path.isdir(models_dir):
    folders = os.listdir(models_dir)
    print(f"\nModel folders: {len(folders)}")
    # Check a few MLmodel files for run_id
    for folder in sorted(folders)[:3]:
        mlmodel_path = os.path.join(models_dir, folder, 'artifacts', 'MLmodel')
        if os.path.exists(mlmodel_path):
            with open(mlmodel_path) as f:
                content = f.read()
            # Search for run_id
            import re
            run_ids = re.findall(r'run_id:\s*(\S+)', content)
            src_run = re.findall(r'source_run_id:\s*(\S+)', content)
            print(f"  {folder}: run_ids={run_ids}, source_run_ids={src_run}")
