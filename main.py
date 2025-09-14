from src.face_recog import FaceRecognition

if __name__ == "__main__":
    fr = FaceRecognition(dataset_dir="dataset")
    fr.build_index()
    label, distance = fr.recognize("is_kenjie.jpg", threshold=0.9)
    print("Result:", label, "| Distance:", distance)
