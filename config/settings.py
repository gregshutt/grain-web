import os
import uuid
from config import storage
from grain.medical import Medical
from distutils.util import strtobool

SECRET_KEY = os.getenv("SECRET_KEY", "")

SERVER_NAME = os.getenv(
    "SERVER_NAME", "localhost:{0}".format(os.getenv("PORT", "8000"))
)

MINIO_HOST = os.getenv("MINIO_HOST", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "openshift-ai-pipeline")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "RN50")
CLIP_CHECKPOINT_PATH = os.getenv("CLIP_CHECKPOINT_PATH", "models/grain/checkpoints/epoch_50.pt")

_MEDICAL_INSTANCE = None

def get_medical():
    global _MEDICAL_INSTANCE
    if _MEDICAL_INSTANCE is None:
        print("Initializing medical instance...")
        temp_path = f"/tmp/{uuid.uuid4()}"
        minio=storage.get_minio_client()
        minio.fget_object(MINIO_BUCKET, CLIP_CHECKPOINT_PATH, temp_path)
        _MEDICAL_INSTANCE = Medical(
            model_name=CLIP_MODEL_NAME, 
            checkpoint_path=temp_path
        )
    return _MEDICAL_INSTANCE