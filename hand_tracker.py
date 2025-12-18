import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# MediaPipe Tasks setup
# -----------------------------
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2
)

# -----------------------------
# Drawing helper
# -----------------------------
def draw_landmarks(image, hand_landmarks):
    h, w, _ = image.shape

    for hand in hand_landmarks:
        # Draw landmarks
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9,10), (10,11), (11,12),     # Middle
            (9,13), (13,14), (14,15), (15,16),    # Ring
            (13,17), (17,18), (18,19), (19,20),   # Pinky
            (0,17)
        ]

        for start, end in connections:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


# -----------------------------
# Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)

        cv2.imshow("Hand Tracking (MediaPipe Tasks)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
