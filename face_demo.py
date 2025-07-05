import numpy as np
import cv2 as cv
import os

# ==== CONFIGURATION ====
IMAGE_PATH = r"Imgs\336635642_bc9fd4bd-de9b-4555-976c-8360576c6708.jpg"
MODEL_PATH = r"models\face_detection_yunet_2023mar.onnx"
SAVE_RESULT = True
SCALE = 1.0
# ========================

def visualize_faces(input_img, faces, fps, thickness=2):
    if faces[1] is not None:
        count = 0
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            score = face[-1]
            label = f"Face {idx+1}"

            cv.rectangle(input_img, (coords[0], coords[1]), 
                         (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.putText(input_img, label, (coords[0], coords[1] - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            count += 1

        print(f"{count} face(s) detected.")
    else:
        print("No face found")
        cv.putText(input_img, "No face found", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.putText(input_img, f'FPS: {fps:.2f}', (1, 16), 
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    # Load image
    img = cv.resize(cv.imread(IMAGE_PATH),(600,600))
    if img is None:
        raise ValueError("Could not load image. Check the path.")

    # Resize if needed
    img_width = int(img.shape[1] * SCALE)
    img_height = int(img.shape[0] * SCALE)
    img = cv.resize(img, (img_width, img_height))

    # Load model
    detector = cv.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000
    )

    # Set input size
    detector.setInputSize((img_width, img_height))

    # Inference
    tm = cv.TickMeter()
    tm.start()
    faces = detector.detect(img)
    tm.stop()

    # Visualize and show
    visualize_faces(img, faces, tm.getFPS())
    cv.imshow("Face Detection Result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save result
    if SAVE_RESULT:
        result_path = os.path.join(os.path.dirname(IMAGE_PATH), "result.jpg")
        cv.imwrite(result_path, img)
        print(f"Saved result image at: {result_path}")

if __name__ == "__main__":
    main()
