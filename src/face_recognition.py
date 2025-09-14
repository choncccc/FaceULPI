import os
import cv2
import numpy as np
import torch
import faiss
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognition:
    """Face recognition with FaceNet embeddings + FAISS indexing"""

    def __init__(self, dataset_dir="dataset", index_path="face_index.bin", labels_path="labels.pkl", device=None):
        # init dataset, device, models, storage
        self.dataset_dir = dataset_dir
        self.index_path = index_path
        self.labels_path = labels_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(image_size=160, margin=20, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.embeddings = []
        self.labels = []  # always a list
        self.index = None

        # try to load saved index
        if os.path.exists(self.index_path) and os.path.exists(self.labels_path):
            self.load_index()
        else:
            print("[INFO] No saved index found. Run build_index() to create one.")

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
        self.embeddings = []
        self.labels = []     

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

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        self.save_index()  # save after building
        print(f"[INFO] Indexed {len(self.labels)} faces and saved to disk")

    def save_index(self):
        # save FAISS index and labels to disk
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.labels_path, "wb") as f:
                pickle.dump(list(self.labels), f)  # force list
            print("[INFO] Index and labels saved")

    def load_index(self):
        # load FAISS index and labels from disk
        self.index = faiss.read_index(self.index_path)
        with open(self.labels_path, "rb") as f:
            self.labels = list(pickle.load(f))  # ensure list
        print(f"[INFO] Loaded index with {len(self.labels)} faces")

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


# Example usage
if __name__ == "__main__":
    fr = FaceRecognition(dataset_dir="dataset")
    fr.build_index()  # run once to (re)build index 
