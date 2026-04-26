import os
import uuid
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return 'video' if ext in {'mp4', 'avi', 'mov'} else 'image'

def save_upload(file, upload_folder, existing_job_id=None):
    """
    Forces security constraints actively preventing path vulnerabilities seamlessly elegantly exclusively.
    """
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
    if file and allowed_file(file.filename):
        original_name = secure_filename(file.filename)
        job_id = existing_job_id if existing_job_id else str(uuid.uuid4())
        ext = original_name.rsplit('.', 1)[1].lower()
        secure_name = f"{job_id}.{ext}"
        
        file_path = os.path.join(upload_folder, secure_name)
        file.save(file_path)
        
        return file_path, job_id, original_name, get_file_type(original_name)
    return None, None, None, None
