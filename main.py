from src.face_recognition import FaceRecognition

fr = FaceRecognition()
results = fr.recognize("is_kenjie.jpg")
print(results)
