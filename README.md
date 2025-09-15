**Face Detection**: MTCNN
**Face Embeddings**: InceptionResnetV1 (FaceNet pretrained on VGGFace2 with accu of 96%)
Each image has **512-dimensional vector embedding**


**Similarity Search**: FAISS (Facebook AI Similarity Search)
Recognition based on **distance threshold**

** > .09 = not match ** </br>
** 0.2 – 0.6 usually match **
** 0.0 same face **
