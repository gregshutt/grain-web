import torch
import faiss
import numpy as np
from PIL import Image
import open_clip
import uuid
from config import settings, storage

class Medical:
    def __init__(self, model_name, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # load the trained model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=checkpoint_path
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.embed_dim = self.model.visual.output_dim
        self.index = faiss.IndexFlatIP(self.embed_dim)
        
        # load clinical notes
        self.images_database = []
        self.notes_database = []

        client = storage.get_minio_client()
        objects = client.list_objects(
            bucket_name=settings.MINIO_BUCKET, prefix="dataset/"
        )

        items = []
        seen_bases = set()
        for obj in objects:
            if obj.object_name.endswith(".txt"):
                continue
            base_name = obj.object_name.rsplit(".", 1)[0]
            if base_name in seen_bases:
                continue
            seen_bases.add(base_name)

            try:
                txt_response = client.get_object(
                    settings.MINIO_BUCKET, f"{base_name}.txt"
                )
                text = txt_response.read().decode("utf-8")
            except:
                text = ""

            if text:
                items.append((obj.object_name, text))
        self.add_cases_to_library([x[0] for x in items], [x[1] for x in items])

    def _get_image_embedding(self, image_path):
        # processes image and returns normalized embedding
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype('float32')

    def add_cases_to_library(self, image_paths, clinical_notes):
        # build the searchable database of cases
        embeddings = []
        for img_path in image_paths:
            temp_path = f"/tmp/{uuid.uuid4()}"
            minio=storage.get_minio_client()
            minio.fget_object(settings.MINIO_BUCKET, img_path, temp_path)
            embeddings.append(self._get_image_embedding(temp_path))
        
        # stack and add to faiss index
        embeddings_matrix = np.vstack(embeddings)
        self.index.add(embeddings_matrix)
        self.images_database.extend(image_paths)
        self.notes_database.extend(clinical_notes)
        print(f"Successfully indexed {len(clinical_notes)} medical cases.")

    def query_diagnosis(self, query_image_path, top_k=3):
        """Finds the most similar historical cases and their notes."""
        query_vec = self._get_image_embedding(query_image_path)
        
        # Search the index
        similarities, indices = self.index.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "confidence_score": float(similarities[0][i]),
                "clinical_note": self.notes_database[idx],
                "image_name": self.images_database[idx],
            })
        return results