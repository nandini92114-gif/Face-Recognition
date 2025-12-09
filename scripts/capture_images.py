# capture_images.pyq
import cv2
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_for_person(name, num_images=30, cam_index=0):
    person_dir = os.path.join(DATASET_DIR, name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open webcam. Check camera index.")
        return

    print(f"Starting capture for {name}. Press 'q' to quit early.")
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.putText(frame, f"{name} - Img {count}/{num_images}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('Capture (press q to quit)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):      # press 'c' to capture
            img_path = os.path.join(person_dir, f"{name}_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print("Saved", img_path)
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    name = input("Enter person name (no spaces): ").strip()
    num = input("Number of images (default 30): ").strip()
    num = int(num) if num.isdigit() else 30
    capture_for_person(name, num_images=num)
