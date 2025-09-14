# src/face_recognition.py
import os
import pickle
import faiss
import numpy as np
import face_recognition


class FaceRecognition:
    def __init__(self, dataset_path="dataset", index_path="face_index.bin", labels_path="labels.pkl"):
        """Initialize with dataset and index paths"""
        self.dataset_path = dataset_path
        self.index_path = index_path
        self.labels_path = labels_path
        self.index = None
        self.labels = []

        # Load index if exists
        if os.path.exists(self.index_path) and os.path.exists(self.labels_path):
            self.load_index()
        else:
            print("No saved index found. Please run build_index() to create one.")

    def build_index(self):
        """Build FAISS index from dataset images"""
        embeddings = []
        labels = []

        for person_name in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)   

                if len(encodings) > 0:
                    embeddings.append(encodings[0])
                    labels.append(person_name)

        if len(embeddings) == 0:
            raise ValueError("No faces found in dataset!")

        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.labels = labels

        self.save_index()  # Save after building
        print("Index built and saved successfully.")

    def save_index(self):
        """Save FAISS index and labels to disk"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.labels_path, "wb") as f:
                pickle.dump(self.labels, f)

    def load_index(self):
        """Load FAISS index and labels from disk"""
        self.index = faiss.read_index(self.index_path)
        with open(self.labels_path, "rb") as f:
            self.labels = pickle.load(f)
        print("Index and labels loaded successfully.")

    def recognize(self, image_path, top_k=1):
        """Recognize face from an input image"""
        if self.index is None:
            raise ValueError("Index not built. Run build_index() first.")

        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            return None

        query = np.array([encodings[0]]).astype("float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((self.labels[idx], dist))
        return results
