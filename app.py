from fastapi import FastAPI, HTTPException
import cv2
from deepface import DeepFace
import uvicorn
import time

app = FastAPI()

models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    "GhostFaceNet",
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'fastmtcnn',
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'centerface',
]

faces = "faces/"  # Path to the face database

# FastAPI endpoint to start the face recognition process
@app.get("/recognize")
# Function to process video feed and continuously detect faces
async def recognize():
    
    cap = cv2.VideoCapture(0)  # Open camera
    

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open camera")
    
    while True:
        time.sleep(1)
        ret, frame = cap.read()  # Capture a frame
        
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to capture image")
        
        try:
            dfs = DeepFace.find(
                img_path=frame,
                db_path=faces,
                model_name=models[0],
                distance_metric = metrics[0],
                detector_backend=backends[0]
            )

            objs = DeepFace.analyze(
                img_path = frame,
                actions = ['emotion'],
                )

            if len(dfs) > 0:
                df = dfs[0]
                obj = objs[0]
                if not df.empty:
                    name = df['identity'].values[0].split("/")[-1][:-4]  # Remove file extension
                    emotion = obj['dominant_emotion']
                    return {'name': name, 'emotion':emotion}

        except Exception as e:
            continue


# Run the FastAPI app if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)