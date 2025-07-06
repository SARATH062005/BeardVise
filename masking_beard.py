import cv2
import mediapipe as mp
import numpy as np
import os

class BeardMasker:
    """
    Detects face and overlays a semi-transparent mask over the beard/mustache area.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        print("BeardMasker initialized with MediaPipe Face Mesh.")

    def apply_beard_mask(self, image, output_image_path: str):
        if image is None:
            print("ERROR: Input image is None.")
            return

        image_height, image_width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("ERROR: No face detected.")
            return

        BEARD_LANDMARK_INDICES = [
            58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
            397, 288, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]

        face_landmarks = results.multi_face_landmarks[0].landmark
        beard_points = np.array([
            [int(face_landmarks[i].x * image_width), int(face_landmarks[i].y * image_height)]
            for i in BEARD_LANDMARK_INDICES
        ], dtype=np.int32)

        hull = cv2.convexHull(beard_points)
        overlay = image.copy()
        cv2.fillConvexPoly(overlay, hull, (0, 255, 0))
        final_image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

        try:
            cv2.imwrite(output_image_path, final_image)
            print(f"Success! Masked image saved to: {output_image_path}")
        except Exception as e:
            print(f"ERROR: Could not save the image. Reason: {e}")


class ImageFaceShapeDetector:
    """Handles detection on a static image from file."""

    def __init__(self, image_path, output_path="masked_from_file.png"):
        self.image_path = image_path
        self.output_path = output_path
        self.masker = BeardMasker()

    def run(self):
        if not os.path.exists(self.image_path):
            print(f"ERROR: File '{self.image_path}' not found.")
            return

        image = cv2.imread(self.image_path)
        if image is None:
            print("ERROR: Failed to load image.")
            return

        print(f"Running detection on: {self.image_path}")
        self.masker.apply_beard_mask(image, self.output_path)


class LiveFaceShapeDetector:
    """Handles detection on an image captured live from webcam."""

    def __init__(self, save_path="captured_image.png", output_path="masked_from_camera1.png"):
        self.save_path = save_path
        self.output_path = output_path
        self.masker = BeardMasker()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot access webcam.")
            return

        print("Press ENTER to capture image. Press ESC to cancel.")
        image = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read from camera.")
                break

            cv2.imshow("Live Camera - Press Enter to Capture", frame)
            key = cv2.waitKey(1)

            if key == 13:  # Enter key
                cv2.imwrite(self.save_path, frame)
                image = frame
                print(f"Image captured and saved to: {self.save_path}")
                break
            elif key == 27:  # ESC key
                print("Capture cancelled.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if image is not None:
            self.masker.apply_beard_mask(image, self.output_path)


# --- Main Function ---
if __name__ == "__main__":
    # Ensure you have an image in a folder named 'Imgs' or change the path
    # For example: r"C:\Users\YourUser\Desktop\my_face.jpg"
    image_path = r"Imgs\portrait-white-man-isolated.jpg"

    # Option 1: Analyze from image file
    # print("--- Running Analysis on Image File ---")
    # image_detector = ImageFaceShapeDetector(image_path)
    # image_detector.run()

    # Option 2: Analyze from camera
    print("\n--- Starting Live Camera Analysis ---")
    live_detector = LiveFaceShapeDetector()
    live_detector.run()
