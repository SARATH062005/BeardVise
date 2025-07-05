import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

# --- File Paths ---
FACE_IMAGE_PATH = 'Imgs\portrait-man-laughing.jpg'
BEARD_IMAGE_PATH = r"Imgs\Beard\beard-png-857.png"
OUTPUT_IMAGE_PATH = 'output_virtual_try_on_scaled_flexible.jpg' # Changed output filename


def virtual_try_on(face_image_path, beard_image_path, scale_multiplier=1.0): # --- MODIFICATION ---
    """
    Overlays a beard image onto a face image using facial landmarks.

    Args:
        face_image_path (str): Path to the face image.
        beard_image_path (str): Path to the beard image (transparent PNG).
        scale_multiplier (float): Factor to adjust the beard size. 1.0 is default.
    """
    # --- 1. Load Images ---
    face_image_cv = cv2.imread(face_image_path)
    if face_image_cv is None:
        print(f"Error: Could not load face image from {face_image_path}")
        return

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

        rgb_image = cv2.cvtColor(face_image_cv, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print("No face detected in the image.")
            return

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = face_image_cv.shape

        # --- 3. Identify Key Anchor Points on the Face ---
        p_philtrum = face_landmarks.landmark[164]
        center_x, center_y = int(p_philtrum.x * w), int(p_philtrum.y * h)
        p_left_jaw = face_landmarks.landmark[57]
        p_right_jaw = face_landmarks.landmark[287]

        # --- 4. Calculate Transformation Parameters ---
        beard_target_width = math.sqrt((p_right_jaw.x * w - p_left_jaw.x * w)**2 + 
                                     (p_right_jaw.y * h - p_left_jaw.y * h)**2)
        
        # --- MODIFICATION START ---
        # Calculate the base scaling factor to fit the jawline
        base_scale_factor = beard_target_width / beard_image_pil.width
        
        # Apply the user-defined multiplier for final size adjustment
        final_scale_factor = base_scale_factor * scale_multiplier
        # --- MODIFICATION END ---
        
        angle_rad = math.atan2(p_right_jaw.y * h - p_left_jaw.y * h, 
                               p_right_jaw.x * w - p_left_jaw.x * w)
        angle_deg = math.degrees(angle_rad)

        # --- 5. Transform the Beard Image ---
        # Use the final_scale_factor for resizing
        new_width = int(beard_image_pil.width * final_scale_factor) # --- MODIFICATION ---
        new_height = int(beard_image_pil.height * final_scale_factor) # --- MODIFICATION ---
        
        # Ensure new dimensions are not zero
        if new_width == 0 or new_height == 0:
            print("Warning: Calculated beard size is zero. Skipping transformation.")
            return

        resized_beard = beard_image_pil.resize((new_width, new_height), Image.LANCZOS)
        rotated_beard = resized_beard.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)

        # --- 6. Overlay the Beard on the Face ---
        paste_x = center_x - rotated_beard.width // 2
        paste_y = center_y - int(new_height * 0.1)

        face_image_pil = Image.fromarray(cv2.cvtColor(face_image_cv, cv2.COLOR_BGR2RGB))
        face_image_pil.paste(rotated_beard, (paste_x, paste_y), rotated_beard)

        # --- 7. Finalize and Display ---
        # --- 7. Finalize and Display ---
        final_image_cv = cv2.cvtColor(np.array(face_image_pil), cv2.COLOR_RGB2BGR)


        cv2.imshow(f"Virtual Beard Try-On (Scale: {scale_multiplier}x)",cv2.resize( final_image_cv, (600, 600)) )
        cv2.imwrite(OUTPUT_IMAGE_PATH, final_image_cv)
        print(f"Successfully created virtual try-on. Saved to {OUTPUT_IMAGE_PATH}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- Run the main function ---
if __name__ == '__main__':
    # --- MODIFICATION ---
    # You can now control the scale from here!
    # 1.0 = default auto-fit size
    # 1.2 = 20% larger
    # 3.0 = 300% larger (as requested)
    
    custom_scale = 1.90
    
    print(f"Applying custom scale multiplier: {custom_scale}")
    virtual_try_on(FACE_IMAGE_PATH, BEARD_IMAGE_PATH, scale_multiplier=custom_scale)