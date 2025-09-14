from src.face_recognition import FaceRecognition

fr = FaceRecognition(dataset_dir="dataset")
label, dist = fr.recognize("not_kenjie.jpg")
print("Result:", label, "| Distance:", dist)