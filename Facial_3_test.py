import cv2
import mediapipe as mp
import numpy as np
import math

# ======================== Core Logic ============================

class FaceShapeAnalyzer:
    def __init__(self, image):
        self.image = cv2.resize(image, (600, 400))
        self.shape = "Unknown"
        self.annotated_image = self.image.copy()

    @staticmethod
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def analyze(self):
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        with mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5) as face_mesh:

            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return self.image, "No Face Detected"

            face_landmarks = results.multi_face_landmarks[0]
            height, width, _ = self.image.shape

            # Key landmark points
            p_left_cheek = (face_landmarks.landmark[234].x * width, face_landmarks.landmark[234].y * height)
            p_right_cheek = (face_landmarks.landmark[454].x * width, face_landmarks.landmark[454].y * height)
            p_left_jaw = (face_landmarks.landmark[172].x * width, face_landmarks.landmark[172].y * height)
            p_right_jaw = (face_landmarks.landmark[397].x * width, face_landmarks.landmark[397].y * height)
            p_forehead = (face_landmarks.landmark[10].x * width, face_landmarks.landmark[10].y * height)
            p_chin = (face_landmarks.landmark[152].x * width, face_landmarks.landmark[152].y * height)

            # Measure distances
            face_width = self.calculate_distance(p_left_cheek, p_right_cheek)
            jaw_width = self.calculate_distance(p_left_jaw, p_right_jaw)
            face_height = self.calculate_distance(p_forehead, p_chin)

            # Ratios
            height_to_width_ratio = face_height / face_width
            jaw_to_face_width_ratio = jaw_width / face_width

            if height_to_width_ratio > 1.4:
                self.shape = "Oval"
            elif height_to_width_ratio < 1.1:
                self.shape = "Square" if jaw_to_face_width_ratio > 0.9 else "Round"
            else:
                self.shape = "Triangle" if jaw_to_face_width_ratio < 0.85 else "Oval"

            # Draw annotations
            mp_drawing.draw_landmarks(
                image=self.annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Debug circles
            for pt in [p_left_cheek, p_right_cheek, p_left_jaw, p_right_jaw, p_forehead, p_chin]:
                cv2.circle(self.annotated_image, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

            # Put labels
            cv2.putText(self.annotated_image, f"Shape: {self.shape}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.annotated_image, f"H/W Ratio: {height_to_width_ratio:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(self.annotated_image, f"J/F Ratio: {jaw_to_face_width_ratio:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            return self.annotated_image, self.shape


# =================== Image From Folder Class =====================

class ImageFaceShapeDetector:
    def __init__(self, image_path):
        self.image_path = image_path

    def run(self):
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Could not load image: {self.image_path}")
            return

        analyzer = FaceShapeAnalyzer(image)
        annotated_image, face_shape = analyzer.analyze()

        print(f"Predicted Face Shape: {face_shape}")
        cv2.imshow("Face Shape Detection (Image)", cv2.resize(annotated_image, (600, 600)))
        cv2.imwrite("output_image_face_shape.jpg", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =================== Live Camera Class =====================

class LiveFaceShapeDetector:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("Error: Camera not accessible")
            return

        print("Press ENTER to capture an image and analyze face shape...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow("Live Feed - Press ENTER to Capture", frame)
            key = cv2.waitKey(1)
            if key == 13:  # Enter key
                print("Image Captured.")
                break
            elif key == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

        analyzer = FaceShapeAnalyzer(frame)
        annotated_image, face_shape = analyzer.analyze()

        print(f"Predicted Face Shape: {face_shape}")
        cv2.imshow("Face Shape Detection (Live)", cv2.resize(annotated_image, (600, 600)))
        cv2.imwrite("output_live_face_shape.jpg", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =================== Main =====================

if __name__ == "__main__":
    # Option 1: Analyze from image file
    image_detector = ImageFaceShapeDetector(r"Imgs\portrait-white-man-isolated.jpg")
    image_detector.run()

    # Option 2: Analyze from camera
    # live_detector = LiveFaceShapeDetector()
    # live_detector.run()
