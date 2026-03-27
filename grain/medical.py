import torch
import faiss
import numpy as np
from PIL import Image
import open_clip
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
        self.notes_database = []

        client = storage.get_minio_client()
        objects = client.list_objects(
            bucket_name=settings.MINIO_BUCKET, prefix="dataset/"
        )
        print(objects)
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
        self.add_cases_to_library(items)

    def _get_image_embedding(self, image_path):
        """Helper to process an image and return a normalized embedding."""
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype('float32')

    def add_cases_to_library(self, image_paths, clinical_notes):
        # build the searchable database of cases
        #embeddings = []
        #for img_path in image_paths:
        #    embeddings.append(self._get_image_embedding(img_path))
        
        # stack and add to faiss index
        #embeddings_matrix = np.vstack(embeddings)
        #self.index.add(embeddings_matrix)
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
                "clinical_note": self.notes_database[idx]
            })
        return results

# --- Example Usage ---
# librarian = MedicalLibrarian('ViT-B-32', 'my_medical_clip.pt')

# 1. Build your "Knowledge Base" (usually done once)
# historical_images = ["case_001.png", "case_002.png"]
# historical_notes = ["Mass in right hilar region...", "Normal cardiac silhouette..."]
# librarian.add_cases_to_library(historical_images, historical_notes)

# 2. Query a new unknown image
# top_cases = librarian.query_diagnosis("patient_x_ray.png", top_k=1)
# print(f"Most likely diagnosis: {top_cases[0]['clinical_note']}")