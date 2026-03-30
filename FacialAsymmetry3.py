import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import pandas as pd

VideoPath = input("Enter Path to Video:")
clean_text = VideoPath.strip('"') 
DrawLandmarks = True

df = pd.read_excel("LandmarkPairs.xlsx")

pairs = [
    (int(row["Left"]), int(row["Right"]))
    for _, row in df.iterrows()
    if not pd.isna(row["Left"]) and not pd.isna(row["Right"])
]

folder_path = "csv_logs_face"
file_name = os.path.basename(clean_text) + "_FacialAsymmetry"

os.makedirs(folder_path, exist_ok=True)
csv_file = os.path.join(folder_path, f"{file_name}.csv")

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "asymmetry value", "minimum", "maximum", "range"])

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(clean_text)


def midpoint(a, b):
    return np.array([
        (a.x + b.x) / 2,
        (a.y + b.y) / 2,
        (a.z + b.z) / 2
    ])


def plane_from_points(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    normal = np.cross(p2 - p1, p3 - p1)
    normal /= np.linalg.norm(normal)

    d = -np.dot(normal, p1)

    if normal[2] < 0:
        normal = -normal
        d = -d

    return normal, d


def signed_distance(point, normal, d):
    return np.dot(normal, point) + d


with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            frame_number += 1

            pointDistances = []

            mid_eye = midpoint(
                face_landmarks.landmark[133],
                face_landmarks.landmark[362]
            )

            mid_mouth = np.array([
                face_landmarks.landmark[168].x,
                face_landmarks.landmark[168].y,
                face_landmarks.landmark[168].z
            ])

            nose_tip = np.array([
                face_landmarks.landmark[6].x,
                face_landmarks.landmark[6].y,
                face_landmarks.landmark[6].z
            ])

            normal, d = plane_from_points(mid_eye, mid_mouth, nose_tip)

            pointDistances = []

            for left_idx, right_idx in pairs:
                left_point = face_landmarks.landmark[left_idx]
                right_point = face_landmarks.landmark[right_idx]

                left = np.array([left_point.x, left_point.y, left_point.z])
                right = np.array([right_point.x, right_point.y, right_point.z])

                midpoint_pair = (left + right) / 2

                distance = abs(signed_distance(midpoint_pair, normal, d))

                pointDistances.append(distance)

            average = np.mean(pointDistances)
            min_val = np.min(pointDistances)
            max_val = np.max(pointDistances)
            range_val = max_val - min_val

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([frame_number, average, min_val, max_val, range_val])

            print(frame_number)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()