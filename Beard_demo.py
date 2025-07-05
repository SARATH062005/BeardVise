import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

# --- File Paths (adjust if your structure is different) ---
FACE_IMAGE_PATH = 'Imgs\portrait-man-laughing.jpg'
BEARD_IMAGE_PATH =  r"Imgs\beard-png-44567.png"
OUTPUT_IMAGE_PATH = 'output_virtual_try_on.jpg'

img = cv2.imread(r"Imgs\beard-png-44567.png")
if img is None:
    print("Error: Could not load beard image.")
else:
    cv2.imshow("Beard Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def virtual_try_on(face_image_path, beard_image_path):
    """
    Overlays a beard image onto a face image using facial landmarks.
    """
    # --- 1. Load Images ---
    # Load the face image with OpenCV
    face_image_cv = cv2.imread(face_image_path)
    if face_image_cv is None:
        print(f"Error: Could not load face image from {face_image_path}")
        return

    # Load the beard image with Pillow to preserve the alpha (transparency) channel
    try:
        beard_image_pil = Image.open(beard_image_path).convert("RGBA")
    except IOError:
        print(f"Error: Could not load beard image from {beard_image_path}")
        return

    # --- 2. Detect Facial Landmarks ---
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        # Convert the BGR image to RGB for MediaPipe
        rgb_image = cv2.cvtColor(face_image_cv, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print("No face detected in the image.")
            return

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = face_image_cv.shape

        # --- 3. Identify Key Anchor Points on the Face ---
        # These landmark indices are chosen to correctly position the beard.
        # Landmark 164 is on the philtrum, a stable center point above the lip.
        # Landmarks 57 and 287 are at the corners of the jawline.
        
        # Center point for the mustache
        p_philtrum = face_landmarks.landmark[164]
        center_x, center_y = int(p_philtrum.x * w), int(p_philtrum.y * h)

        # Points for scaling and rotating the beard
        p_left_jaw = face_landmarks.landmark[57]
        p_right_jaw = face_landmarks.landmark[287]

        # --- 4. Calculate Transformation Parameters ---
        # Calculate the width the beard should be on the face
        beard_target_width = math.sqrt((p_right_jaw.x * w - p_left_jaw.x * w)**2 + 
                                     (p_right_jaw.y * h - p_left_jaw.y * h)**2)
        
        # Calculate the scaling factor
        scale_factor = beard_target_width / beard_image_pil.width
        
        # Calculate the angle of the jawline to rotate the beard
        angle_rad = math.atan2(p_right_jaw.y * h - p_left_jaw.y * h, 
                               p_right_jaw.x * w - p_left_jaw.x * w)
        angle_deg = math.degrees(angle_rad)

        # --- 5. Transform the Beard Image ---
        # Resize the beard
        new_width = int(beard_image_pil.width * scale_factor)
        new_height = int(beard_image_pil.height * scale_factor)
        resized_beard = beard_image_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # Rotate the beard
        rotated_beard = resized_beard.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)

        # --- 6. Overlay the Beard on the Face ---
        # Calculate the top-left position to paste the rotated beard
        # We adjust the center point to align the top-center of the beard image with our anchor
        paste_x = center_x - rotated_beard.width // 2
        paste_y = center_y - int(new_height * 0.1) # Minor adjustment to position it perfectly under the nose

        # Convert OpenCV face image to Pillow format
        face_image_pil = Image.fromarray(cv2.cvtColor(face_image_cv, cv2.COLOR_BGR2RGB))
        
        # Paste the beard onto the face using the alpha channel as a mask
        face_image_pil.paste(rotated_beard, (paste_x, paste_y), rotated_beard)

        # --- 7. Finalize and Display ---
        # Convert back to OpenCV format to display and save
        final_image_cv = cv2.cvtColor(np.array(face_image_pil), cv2.COLOR_RGB2BGR)

        # Display the result
        cv2.imshow("Virtual Beard Try-On",cv2.resize( final_image_cv, (600, 600)))
        cv2.imwrite(OUTPUT_IMAGE_PATH, final_image_cv)
        print(f"Successfully created virtual try-on. Saved to {OUTPUT_IMAGE_PATH}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- Run the main function ---
if __name__ == '__main__':
    virtual_try_on(FACE_IMAGE_PATH, BEARD_IMAGE_PATH)