import cv2
import mediapipe as mp
import urllib.request
import os
import time

# --------------------------------------------------
# Download MediaPipe Hand Landmarker model if needed
# --------------------------------------------------

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")


# --------------------------------------------------
# MediaPipe Hand Landmarker
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)


# --------------------------------------------------
# Open Webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()


# --------------------------------------------------
# Hand Connections
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------------------------
# Start Camera
# --------------------------------------------------

start_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        print("Error: Cannot read webcam frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Timestamp
    timestamp_ms = int((time.time() - start_time) * 1000)

    # Detect hands
    results = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------
    # Check detected hands
    # --------------------------------------------------

    if results.hand_landmarks:

        for hand_index, hand_landmarks in enumerate(results.hand_landmarks):

            # Get hand name
            hand_name = results.handedness[hand_index][0].category_name

            # --------------------------------------------------
            # Draw hand landmarks
            # --------------------------------------------------

            height, width, _ = frame.shape

            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 255, 255),
                    -1
                )

            # Draw connections
            for start, end in HAND_CONNECTIONS:

                if start < len(points) and end < len(points):

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 255, 255),
                        2
                    )


            # --------------------------------------------------
            # Finger detection
            # --------------------------------------------------

            lm = hand_landmarks

            # Thumb
            if hand_name == "Right":
                thumb = lm[4].x < lm[3].x
            else:
                thumb = lm[4].x > lm[3].x

            # Other four fingers
            index = lm[8].y < lm[6].y
            middle = lm[12].y < lm[10].y
            ring = lm[16].y < lm[14].y
            pinky = lm[20].y < lm[18].y


            # Store finger status
            fingers = {
                "Thumb": thumb,
                "Index": index,
                "Middle": middle,
                "Ring": ring,
                "Pinky": pinky
            }


            # Total fingers
            total = sum(fingers.values())


            # --------------------------------------------------
            # Display finger status
            # --------------------------------------------------

            y = 40 + (hand_index * 220)

            cv2.putText(
                frame,
                f"{hand_name} Hand",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            y += 35


            for finger, raised in fingers.items():

                status = "UP" if raised else "DOWN"

                if raised:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"{finger}: {status}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                y += 30


            # --------------------------------------------------
            # Display total
            # --------------------------------------------------

            cv2.putText(
                frame,
                f"Total: {total}",
                (20, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )


    # --------------------------------------------------
    # Show camera
    # --------------------------------------------------

    cv2.imshow(
        "Individual Finger Counter",
        frame
    )


    # --------------------------------------------------
    # Press Q to exit
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Release resources
# --------------------------------------------------

cap.release()
detector.close()
cv2.destroyAllWindows()