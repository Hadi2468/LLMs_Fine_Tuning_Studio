import json
from pathlib import Path

JOB_DIR = Path(r"G:\My Drive\LLMs_studio\jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)


def submit_job(config: dict):

    job_id = config["job_id"]

    job = {
        "job_id": job_id,
        "status": "pending",
        "config": config
    }

    job_path = JOB_DIR / f"{job_id}.json"

    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)

    print(f"✅ Job submitted: {job_id}")
    print(f"📁 Saved to: {job_path}")

    return job_id