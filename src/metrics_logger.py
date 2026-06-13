import json
from pathlib import Path
from datetime import datetime


def save_train_metrics(job_id: str, metrics: dict, base_dir: str = "."):
    """
    Save training metrics into local logs directory.
    
    Args:
        job_id: unique training job id
        metrics: dictionary of training metrics
        base_dir: project root directory
    """

    log_dir = Path(base_dir) / "logs" / job_id
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = log_dir / "train_metrics.json"

    payload = {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics
    }

    with open(metrics_file, "w") as f:
        json.dump(payload, f, indent=4)

    print(f"✅ Metrics saved at: {metrics_file}")