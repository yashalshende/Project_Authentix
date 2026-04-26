from threading import Lock

progress_store = {}
store_lock = Lock()

def set_progress(job_id, percentage, message):
    with store_lock:
        if job_id not in progress_store:
             progress_store[job_id] = {}
        progress_store[job_id]["percentage"] = percentage
        progress_store[job_id]["message"] = message
        progress_store[job_id]["complete"] = False

def get_progress(job_id):
    with store_lock:
        return progress_store.get(job_id, {"percentage": 0, "message": "Initializing neural uplink...", "complete": False})

def complete_progress(job_id):
    with store_lock:
        if job_id in progress_store:
            progress_store[job_id]["percentage"] = 100
            progress_store[job_id]["message"] = "Analysis Complete."
            progress_store[job_id]["complete"] = True
