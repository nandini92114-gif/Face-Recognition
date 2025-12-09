import face_recognition
import os
import pickle
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
ENC_DIR = BASE_DIR / "encodings"
ENC_DIR.mkdir(exist_ok=True)

if not DATASET_DIR.exists():
    print(f"Dataset directory not found at {DATASET_DIR}. Run capture_images.py first to create it.")
    sys.exit(1)

encodings_file = ENC_DIR / "encodings.pkl"

known_encodings = []
known_names = []

for person_dir in DATASET_DIR.iterdir():
    if not person_dir.is_dir(): 
        continue
    name = person_dir.name
    for img_path in person_dir.glob("*"):
        image = face_recognition.load_image_file(str(img_path))
        boxes = face_recognition.face_locations(image)
        if len(boxes) == 0:
            print(f"No face found in {img_path}")
            continue
        enc = face_recognition.face_encodings(image, boxes)[0]
        known_encodings.append(enc)
        known_names.append(name)
        print("Encoded:", img_path)

data = {"encodings": known_encodings, "names": known_names}

with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print("Saved encodings to:", encodings_file)
