import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import winsound  # For alarm sound (Windows)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indices for MediaPipe (different from Dlib!)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # 6 points
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # 6 points

# Thresholds
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 20

counter = 0
drowsy = False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get landmark coordinates
            mesh_points = np.array([(int(p.x * w), int(p.y * h)) for p in face_landmarks.landmark])

            # Draw multiple points
            for point in mesh_points:
                cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            # Extract eye points
            left_eye_pts = mesh_points[LEFT_EYE]
            right_eye_pts = mesh_points[RIGHT_EYE]

            # EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            cv2.polylines(frame, [left_eye_pts], True, (255, 0, 0), 1)
            cv2.polylines(frame, [right_eye_pts], True, (255, 0, 0), 1)

            # Drowsiness detection
            if ear < EAR_THRESH:
                counter += 1
                if counter >= EAR_CONSEC_FRAMES:
                    if not drowsy:  # Play sound only once per detection
                        winsound.Beep(2500, 1000)  # Frequency 2500 Hz, duration 1 sec
                    drowsy = True
                    cv2.putText(frame, "DROWSY ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                counter = 0
                drowsy = False

            # Show EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Drowsiness Detection - MediaPipe", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q'
        break

cap.release()
cv2.destroyAllWindows()
