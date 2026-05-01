import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

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
    
    # Process hand detection
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks and get index finger tip
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates (landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, c = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Draw circle at index finger tip
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            
            # Check if finger is up (for drawing control)
            # Compare y-coordinate of index finger tip with middle finger tip (landmark 12)
            middle_finger_tip = hand_landmarks.landmark[12]
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
