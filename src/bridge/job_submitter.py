import json
import time
from pathlib import Path

from src.config import DATA_PATH


JOB_DIR = Path(r"G:\My Drive\LLMs_studio\jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)


def submit_job(config: dict):
    job_id = f"job_{int(time.time())}"

    job = {
        "job_id": job_id,
        "status": "pending",
        "config": config
    }

    job_path = JOB_DIR / f"{job_id}.json"

    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)

    print(f"✅ Job submitted: {job_id}")
    return job_id