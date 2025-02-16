import os
os.environ["BLINKA_JETSON_I2C_BUS"] = "7"
os.environ["BLINKA_FORCEBOARD"] = "JETSON_ORIN_NX"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys, select, tty, termios
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import time
from yapper import Yapper, PiperSpeaker, PiperVoiceUS, PiperQuality

# For OLED display
try:
    import board
    import busio
    import adafruit_ssd1306
except Exception as e:
    print("OLED library import error:", e)

# -------------------------
# Configuration Flags
# -------------------------
ENABLE_OLED = True

# -------------------------
# Set up OLED Display (if enabled)
# -------------------------
if ENABLE_OLED:
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        WIDTH = 64
        HEIGHT = 48
        display = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3d)
        display.fill(0)
        display.show()
    except Exception as e:
        print("OLED initialization error:", e)
        ENABLE_OLED = False

# -------------------------
# TTS Setup
# -------------------------
piper = PiperSpeaker(voice=PiperVoiceUS.BRYCE)
piper.say("hello")
tts_engine = Yapper(speaker=piper)

# -------------------------
# Define class names (order must match training)
# -------------------------
class_names = [
    "A", "B", "C", "D", "delete", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
]

# -------------------------
# Load the ONNX model
# -------------------------
onnx_model_path = "asl_classifier.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# -------------------------
# Setup MediaPipe Hands
# -------------------------
mp_hands = mp.solutions.hands

# -------------------------
# Video Capture (no visualization)
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

def preprocess_landmarks(hand_landmarks):
    coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    flat = np.array(coords, dtype=np.float32).flatten()
    return flat.reshape(1, -1)

threshold = 0.98  # Confidence threshold

word_buffer = ""
debounce_interval = 2  # seconds after each commit
last_commit_time = 0

paused = False  # Pause toggle

def update_display(buffer_text, paused_status):
    """Update the OLED display with the current word buffer and status, if enabled.
       The text is drawn inverted (from the bottom) so that if the display is mounted upside down,
       the text appears right-side up."""
    if not ENABLE_OLED:
        return
    try:
        display.fill(0)
        # Draw text such that it appears right-side up when the display is mounted upside down.
        display.text("Buffer:", 0, 18, 1)
        disp_text = buffer_text if len(buffer_text) <= 10 else buffer_text[-10:]
        display.text(disp_text, 0, 8, 1)
        status_text = "PAUSED" if paused_status else "RUNNING"
        display.text(status_text, 0, 0, 1)
        display.show()
    except Exception as e:
        print("OLED display error:", e)

def get_keypress(timeout=0.01):
    """Return a single character from stdin if available, else None."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None

print("Starting inference (no visualization).")
print("Press space to toggle pause; press ESC to quit.")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while True:
        # Check for key press via stdin
        key = get_keypress()
        if key:
            if key == "\x1b":  # ESC key
                break
            elif key == " ":
                paused = not paused
                if paused:
                    print("Paused.")
                else:
                    print("Resumed.")
                time.sleep(0.3)  # debounce delay
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process frame for hand landmarks (no GUI visualization)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True  # Not used for display

        if not paused:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                input_tensor = preprocess_landmarks(hand_landmarks)
                
                ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
                ort_outs = ort_session.run(None, ort_inputs)
                logits = ort_outs[0]
                
                # Compute softmax probabilities
                if logits.ndim == 1:
                    probabilities = np.exp(logits) / np.sum(np.exp(logits))
                else:
                    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                    probabilities = probabilities[0]
                
                pred_idx = int(np.argmax(probabilities))
                predicted_letter = class_names[pred_idx]
                max_conf = probabilities[pred_idx]
                
                if max_conf >= threshold:
                    current_time = time.time()
                    if current_time - last_commit_time >= debounce_interval:
                        if predicted_letter == "space":
                            if word_buffer:
                                print("Speaking word:", word_buffer)
                                tts_engine.yap(word_buffer, plain=True)
                                word_buffer = ""
                            else:
                                print("Space detected, but buffer is empty.")
                        elif predicted_letter == "delete":
                            if word_buffer:
                                word_buffer = word_buffer[:-1]
                                print("Letter deleted. Current buffer:", word_buffer)
                            else:
                                print("Delete detected, but buffer is empty.")
                        else:
                            word_buffer += predicted_letter
                            print("Current buffer:", word_buffer)
                        last_commit_time = current_time
            else:
                print("No hand detected.")

        # Update OLED display (if enabled)
        update_display(word_buffer, paused)
        time.sleep(0.1)

cap.release()
