import cv2
from deepface import DeepFace

class FaceRecognition:
    def __init__(self):
        self.models = [
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

        self.metrics = ["cosine", "euclidean", "euclidean_l2"]
        
        self.backends = [
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
        
        self.faces = "faces/"
        self.cap = cv2.VideoCapture(0)

    def get_name(self, frame):
        try:
            # Find the face in the frame
            dfs = DeepFace.find(
                img_path=frame,
                db_path=self.faces,
                model_name=self.models[0],
                distance_metric=self.metrics[0],
                detector_backend=self.backends[0]
            )
            
            if len(dfs) > 0:
                df = dfs[0]
                name = df['identity']  # Return the name
                return name
            return None

        except Exception as e:
            print(f"Error in get_name: {e}")
            return None

    def get_emotion(self, frame):
        try:
            # Analyze the emotion in the frame
            objs = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
            )

            if len(objs) > 0:
                obj = objs[0]
                emotion = obj['dominant_emotion']  # Return the dominant emotion
                return emotion
            return None

        except Exception as e:
            print(f"Error in get_emotion: {e}")
            return None

    def process_frame(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                return frame

        self.cap.release()
        cv2.destroyAllWindows()

# if __name__ == '__main__':
#     FaceRecognition().process_frame()