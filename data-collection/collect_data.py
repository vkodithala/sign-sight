import cv2
import mediapipe as mp
import time
import pickle
import os

# Define the classes we want to collect data for.
classes_to_collect = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "space",
    "delete"
]

num_samples = 300

# Define the output file name.
output_file = "collected_asl_data.pkl"

if os.path.exists(output_file):
    with open(output_file, "rb") as f:
        collected_data = pickle.load(f)
    # Ensure all classes are present in the dictionary.
    for cls in classes_to_collect:
        if cls not in collected_data:
            collected_data[cls] = []
    print("Existing data loaded. Sample counts per class:")
    for cls in classes_to_collect:
        count = len(collected_data.get(cls, []))
        print(f"  {cls}: {count} samples")
else:
    collected_data = {cls: [] for cls in classes_to_collect}
    print("No existing data found. Starting fresh.")


# Setup MediaPipe Hands and drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # We only need one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Start webcam capture.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


def get_landmarks(frame):
    """
    Process the frame with MediaPipe Hands and return the landmarks (x, y) for the first hand.
    Returns None if no hand is detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        # Return the first detected hand's landmarks.
        return results.multi_hand_landmarks[0], results
    else:
        return None, results


print("Starting data collection. Press Ctrl+C to quit.")
try:
    while True:
        # Prompt the user to enter a class to record data for.
        target_class = (
            input(f"Enter class to record {classes_to_collect} (or 'quit' to quit): ")
            .strip()
            .lower()
        )
        if target_class == "quit":
            break
        if target_class not in classes_to_collect:
            print("Invalid class. Allowed keys:", classes_to_collect)
            continue

        print(f"Recording data for class '{target_class}'.")
        print(f"Collecting {num_samples} samples at approx 10 samples per second...")
        samples_collected = 0

        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            # Get landmarks and results from MediaPipe.
            hand_landmarks, results = get_landmarks(frame)

            # If landmarks detected, draw them on the frame.
            if hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                # Extract (x,y) coordinates.
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                collected_data[target_class].append(landmarks)
                samples_collected += 1

            # Overlay progress information on the frame.
            progress_text = (
                f"Class: '{target_class}' | Sample: {samples_collected}/{num_samples}"
            )
            cv2.putText(
                frame,
                progress_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Display the frame.
            cv2.imshow("Data Collection", frame)

            # Wait a little: roughly 10 frames per second.
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to break early from current recording
                break
            time.sleep(0.05)

        print(f"Finished collecting data for class '{target_class}'.\n")

except KeyboardInterrupt:
    print("\nData collection interrupted by user.")

# Cleanup.
cap.release()
cv2.destroyAllWindows()
hands.close()

# Save the collected data to a pickle file.
with open(output_file, "wb") as f:
    pickle.dump(collected_data, f)
print(f"Data saved to {output_file}")
