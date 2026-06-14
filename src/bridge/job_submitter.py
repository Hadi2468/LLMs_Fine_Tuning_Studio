import json
from pathlib import Path
from src.config import GOOGLE_DRIVE_PATH

JOB_DIR = GOOGLE_DRIVE_PATH["jobs"]
JOB_DIR.mkdir(parents=True, exist_ok=True)


def submit_job(config: dict):

    job_id = config["job_id"]

    job = {
        "job_id": job_id,
        "status": "pending",
        "config": config
    }

    print("JOB_DIR =", JOB_DIR)
    print("JOB_DIR exists =", JOB_DIR.exists())

    job_path = JOB_DIR / f"{job_id}.json"

    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)

    print(f"✅ Job submitted: {job_id}")
    print(f"📁 Saved to: {job_path}")

    return job_id