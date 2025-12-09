import face_recognition
import cv2
import pickle
import csv
from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
ENC_FILE = BASE_DIR / "encodings" / "encodings.pkl"
ATT_FILE = BASE_DIR / "attendance.csv"

TOLERANCE = 0.45
FRAME_RESIZE = 0.25

if not ENC_FILE.exists():
    print(f"Encodings file not found at {ENC_FILE}. Run encode_faces.py first to generate it.")
    sys.exit(1)

with open(ENC_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

marked = set()

if not ATT_FILE.exists():
    with open(ATT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    now = datetime.now()
    with open(ATT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    print("Marked:", name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam. Check that a camera is connected and not used by another application.")
    sys.exit(1)

print("Starting recognition... press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0,0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
        name = "Unknown"

        if True in matches:
            i = matches.index(True)
            name = known_names[i]

        top *= int(1 / FRAME_RESIZE)
        right *= int(1 / FRAME_RESIZE)
        bottom *= int(1 / FRAME_RESIZE)
        left *= int(1 / FRAME_RESIZE)

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if name != "Unknown" and name not in marked:
            mark_attendance(name)
            marked.add(name)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
