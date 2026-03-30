import cv2
import mediapipe as mp
import numpy as np
import math
import os
import csv

VideoPath = input("Enter Path to Video:")
clean_text = VideoPath.strip('"')

folder_path = "csv_logs_face"
file_name = os.path.basename(clean_text) + "_PointTracking"
print(file_name)

os.makedirs(folder_path, exist_ok=True)
csv_file = os.path.join(folder_path, f"{file_name}.csv")

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame",
        "change 1.x", "change 1.y", "change 1.z",
        "change 2.x", "change 2.y", "change 2.z",
        "change 3.x", "change 3.y", "change 3.z",
        "change 4.x", "change 4.y", "change 4.z",
        "change 5.x", "change 5.y", "change 5.z",
        "change 6.x", "change 6.y", "change 6.z"
    ])

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(clean_text)

points = [221, 441, 1, 62, 292, 152]

REF_A = 4
REF_B = 6


def lm(face_landmarks, idx):
    p = face_landmarks.landmark[idx]
    return np.array([p.x, p.y, p.z])


def get_head_transform(face_landmarks):
    a = lm(face_landmarks, REF_A)
    b = lm(face_landmarks, REF_B)
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    return a, angle


def to_head_local(point, origin, angle):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    dz = point[2] - origin[2]

    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy

    return (lx, ly, dz)


with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    frame_number = 0
    old_vals = [(0.0, 0.0, 0.0)] * 6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            frame_number += 1

            origin, angle = get_head_transform(face_landmarks)

            new_points = [
                to_head_local(lm(face_landmarks, p), origin, angle)
                for p in points
            ]

            if frame_number == 1:
                old_vals = new_points
                continue

            changes = [
                (
                    abs(new_points[i][0] - old_vals[i][0]),
                    abs(new_points[i][1] - old_vals[i][1]),
                    abs(new_points[i][2] - old_vals[i][2]),
                )
                for i in range(6)
            ]

            old_vals = new_points

            row = [frame_number]
            for dx, dy, dz in changes:
                row.extend([dx, dy, dz])

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(frame_number)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()