import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import time
from yapper import Yapper, PiperSpeaker, PiperVoiceUS, PiperQuality

# Define class names (order must match training)
class_names = [
    "A", "B", "C", "D", "delete", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"
]

# Load the ONNX model
onnx_model_path = "asl_classifier.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands

piper = PiperSpeaker(voice=PiperVoiceUS.BRYCE)
piper.say("hello")
tts_engine = Yapper(speaker=piper)

# Start webcam capture.
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
# Debounce interval (seconds) after each detected letter
debounce_interval = 2  
last_commit_time = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    
    print("Starting inference. Press Ctrl+C to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        
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
                # Check if enough time has passed since the last commit
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
                            word_buffer = word_buffer[:-1]  # Remove the last character
                            print("Letter deleted. Current buffer:", word_buffer)
                        else:
                            print("Delete detected, but buffer is empty.")
                    else:
                        word_buffer += predicted_letter
                        print("Current buffer:", word_buffer)
                    last_commit_time = current_time
        else:
            print("No hand detected.")

        # Small sleep to prevent high CPU usage
        time.sleep(0.1)

cap.release()
