from src.face_recognition import FaceRecognition

fr = FaceRecognition(dataset_dir="dataset")
label, dist = fr.recognize("test_images/is_kenjie2flip.jpg")
print("Result:", label, "| Distance:", dist)