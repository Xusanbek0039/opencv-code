import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands with the latest API
try:
    # Try the new tasks API first
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Create hand landmarker with default model
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=None),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
    hand_landmarker = HandLandmarker.create_from_options(options)
    use_new_api = True
except:
    # Fallback to older API if available
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        use_new_api = False
    except:
        print("MediaPipe hand tracking not available. Please install a compatible version.")
        exit(1)

# Initialize camera
cap = cv2.VideoCapture(0)

# Create canvas
ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)
else:
    canvas = None

# Previous finger position
prev_x, prev_y = 0, 0
drawing = False

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get frame dimensions
    h, w, c = frame.shape
    
    # Convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process hand detection
    results = hand_landmarker.detect(mp_image)
    
    # Draw hand landmarks and get index finger tip
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Draw hand skeleton manually
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17)  # Palm
            ]
            for connection in connections:
                start_idx, end_idx = connection
                start_point = hand_landmarks[start_idx]
                end_point = hand_landmarks[end_idx]
                start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Draw landmarks
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
            # Get index finger tip coordinates (landmark 8)
            index_finger_tip = hand_landmarks[8]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Draw circle at index finger tip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            
            # Check if finger is up (for drawing control)
            # Compare y-coordinate of index finger tip with middle finger tip (landmark 12)
            middle_finger_tip = hand_landmarks[12]
            if index_finger_tip.y < middle_finger_tip.y:
                # Index finger is up - enable drawing
                if drawing and prev_x != 0 and prev_y != 0:
                    # Draw line from previous position to current position
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                drawing = True
            else:
                # Index finger is down - disable drawing
                drawing = False
            
            # Update previous position
            prev_x, prev_y = x, y
    else:
        # No hand detected
        drawing = False
        prev_x, prev_y = 0, 0
    
    # Combine frame and canvas
    if canvas is not None:
        frame = cv2.bitwise_or(frame, canvas)
    
    # Display instructions
    cv2.putText(frame, "Press 'C' to clear, 'ESC' to exit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Point index finger up to draw", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show result
    cv2.imshow("Hand Drawing", frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):  # C key to clear canvas
        if canvas is not None:
            canvas = np.zeros_like(frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
