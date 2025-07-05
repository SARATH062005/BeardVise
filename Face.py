import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# --- Main Code ---

# Load the image file
image_path = 'Imgs\42113399_8983861.jpg'
# image = cv2.resize(cv2.imread(image_path),(500,500))
image =cv2.imread(image_path)

if image is None:
    print(f"Error: Could not read image from {image_path}")
else:
    # MediaPipe works with RGB images, but OpenCV loads them as BGR. So, we convert.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the face detector
    # min_detection_confidence: 0.5 means it will consider detections with at least 50% confidence.
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        
        # Perform face detection on the image
        results = face_detection.process(rgb_image)

        # Check if any faces were detected
        if results.detections:
            print(f"Found {len(results.detections)} face(s).")
            
            # Loop through each detected face
            for detection in results.detections:
                # The 'detection' object contains the bounding box and keypoints.
                # We can use a handy drawing utility from MediaPipe to draw it.
                mp_drawing.draw_detection(image, detection)
                
                # To draw manually (for more control):
                # bboxC = detection.location_data.relative_bounding_box
                # ih, iw, _ = image.shape
                # bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                #        int(bboxC.width * iw), int(bboxC.height * ih)
                # cv2.rectangle(image, bbox, (255, 0, 255), 2)

        else:
            print("No face detected in the image.")

    # Display the final image with the bounding box
    cv2.imshow('Face Detection - MediaPipe', image)

    # Wait until a key is pressed to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()