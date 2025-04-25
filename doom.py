import cv2
import mediapipe as mp
import subprocess
import time

# Setup mediapipe face mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)

# Landmark indices
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
NOSE_TIP = 1

# Thresholds
SWIPE_UP_THRESHOLD = -15   # Head tilt up
SWIPE_DOWN_THRESHOLD = 15  # Head tilt down
SMILE_WIDTH_THRESHOLD = 60  # Adjust based on camera resolution

# Cooldown setup
cooldown = 2
last_action = time.time() - cooldown
initial_nose_y = None

def double_tap():
    print("😊 Smile detected - Double Tapping!")
    x, y = 500, 1000
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])
    time.sleep(0.1)
    subprocess.run(['adb', 'shell', 'input', 'tap', str(x), str(y)])

def swipe_up():
    print("⬆️ Swiping up...")
    subprocess.run(['adb', 'shell', 'input', 'swipe', '500', '1500', '500', '500'])

def swipe_down():
    print("⬇️ Swiping down...")
    subprocess.run(['adb', 'shell', 'input', 'swipe', '500', '500', '500', '1500'])

def reset_initial_position():
    global initial_nose_y
    initial_nose_y = None
    print("🔄 Calibration reset!")

# Main loop
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Key press events - 'r' resets calibration, 'Esc' exits the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # Calibration reset button
        reset_initial_position()
    if key == 27:  # Esc key to exit
        break

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # --- SMILE DETECTION ---
        left_x = int(landmarks[MOUTH_LEFT].x * w)
        right_x = int(landmarks[MOUTH_RIGHT].x * w)
        mouth_width = abs(right_x - left_x)

        cv2.putText(frame, f"Mouth Width: {mouth_width}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if mouth_width > SMILE_WIDTH_THRESHOLD and (time.time() - last_action) > cooldown:
            double_tap()
            last_action = time.time()

        # --- FACE TILT DETECTION ---
        nose_y = int(landmarks[NOSE_TIP].y * h)
        if initial_nose_y is None:
            initial_nose_y = nose_y

        NEUTRAL_BUFFER = 10
        delta_y = nose_y - initial_nose_y
        cv2.putText(frame, f'Nose delta Y: {delta_y}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if delta_y < -NEUTRAL_BUFFER and abs(delta_y) > abs(SWIPE_UP_THRESHOLD) and (time.time() - last_action) > cooldown:
            swipe_up()
            last_action = time.time()
        elif delta_y > NEUTRAL_BUFFER and delta_y > SWIPE_DOWN_THRESHOLD and (time.time() - last_action) > cooldown:
            swipe_down()
            last_action = time.time()

    # Display calibration instruction on screen
    cv2.putText(frame, "Press 'r' for calibration reset", (30, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Face Control", frame)

cap.release()
cv2.destroyAllWindows()
