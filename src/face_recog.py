import os
import cv2
import numpy as np
import torch
import faiss
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognition:
    """Face recognition with FaceNet embeddings + FAISS indexing"""

    def __init__(self, dataset_dir="dataset", device=None):
        # init dataset, device, models, storage
        self.dataset_dir = dataset_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(image_size=160, margin=20, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embeddings = []
        self.labels = []
        self.index = None

    def _extract_label(self, filename):
        # extract label (person name) from filename
        return filename.split("_")[0]

    def _load_image(self, path):
        # load image and convert to RGB
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def build_index(self):
        # build FAISS index from dataset images
        valid_exts = (".jpg", ".jpeg", ".png")
        for file in os.listdir(self.dataset_dir):
            if not file.lower().endswith(valid_exts):
                continue
            path = os.path.join(self.dataset_dir, file)

            label = self._extract_label(file)
            img = self._load_image(path)
            if img is None:
                continue

            face = self.detector(img)
            if face is None:
                continue

            with torch.no_grad():
                emb = self.embedder(face.unsqueeze(0).to(self.device)).cpu().numpy()[0]

            self.embeddings.append(emb)
            self.labels.append(label)

        self.embeddings = np.array(self.embeddings).astype("float32")
        self.labels = np.array(self.labels)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        print(f"[INFO] Indexed {len(self.labels)} faces")

    def recognize(self, img_path, threshold=0.9):
        # recognize face in an input image
        img = self._load_image(img_path)
        if img is None:
            return None, None

        face = self.detector(img)
        if face is None:
            return None, None

        with torch.no_grad():
            query_emb = self.embedder(face.unsqueeze(0).to(self.device)).cpu().numpy().astype("float32")

        dist, idx = self.index.search(query_emb, k=1)
        if dist[0][0] < threshold:
            return self.labels[idx[0][0]], dist[0][0]
        else:
            return "Unknown", dist[0][0]
