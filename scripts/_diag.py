import os, mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("indonesian-stock-prediction")

# Collect all run_ids from MLmodel files on disk
models_dir = "mlruns/1/models"
mlmodel_ids = set()
for m in os.listdir(models_dir):
    p = os.path.join(models_dir, m, "artifacts", "MLmodel")
    if os.path.exists(p):
        for line in open(p):
            if "run_id:" in line:
                mlmodel_ids.add(line.split(":")[1].strip())

print(f"Models on disk: {len(mlmodel_ids)}")

# Check first 10 tickers
tickers = ["AADI.JK","ADMR.JK","ADRO.JK","AKRA.JK","AMMN.JK",
           "AMRT.JK","ANTM.JK","ARTO.JK","ASII.JK","BBCA.JK"]
for t in tickers:
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"params.ticker = '{t}'",
        order_by=["metrics.avg_roc_auc DESC"],
        max_results=1,
    )
    if runs:
        rid = runs[0].info.run_id
        auc = runs[0].data.metrics.get("avg_roc_auc", "?")
        on_disk = rid in mlmodel_ids
        print(f"{t}: {rid[:12]}... AUC={auc} on_disk={on_disk}")
    else:
        print(f"{t}: NO RUNS FOUND")
