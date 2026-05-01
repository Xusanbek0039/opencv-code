import cv2
import numpy as np
import random

# Create a simulated video feed (no camera required)
def create_demo_frame(width=640, height=480):
    # Create a colorful background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(height):
        frame[i] = [i // 3, 50, 100 + i // 4]
    
    # Add some text
    cv2.putText(frame, "DEMO MODE - Move mouse to draw", (width//2 - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Initialize
width, height = 640, 480
canvas = np.zeros((height, width, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0
drawing = False
demo_counter = 0

print("Demo Drawing Application")
print("Click and drag mouse to draw")
print("Press 'C' to clear canvas")
print("Press 'ESC' to exit")

while True:
    # Create demo frame
    frame = create_demo_frame(width, height)
    
    # Add animated elements
    demo_counter += 1
    circle_x = int(width//2 + 100 * np.sin(demo_counter * 0.05))
    circle_y = int(height//2 + 50 * np.cos(demo_counter * 0.03))
    cv2.circle(frame, (circle_x, circle_y), 20, (0, 255, 255), -1)
    
    # Simulate hand position with mouse
    # In a real application, this would be the detected hand position
    mouse_x, mouse_y = -1, -1
    
    # Get mouse position (simulating hand tracking)
    # We'll use a moving point as "hand" for demo
    hand_x = int(width//2 + 150 * np.sin(demo_counter * 0.02))
    hand_y = int(height//2 + 100 * np.cos(demo_counter * 0.03))
    
    # Draw "hand" indicator
    cv2.circle(frame, (hand_x, hand_y), 15, (0, 255, 0), -1)
    cv2.circle(frame, (hand_x, hand_y), 20, (0, 255, 0), 2)
    
    # Simulate drawing when "hand" is in upper half
    if hand_y < height // 2:
        if drawing and prev_x != 0 and prev_y != 0:
            cv2.line(canvas, (prev_x, prev_y), (hand_x, hand_y), (255, 0, 0), 5)
        drawing = True
        status_text = "Drawing: ON"
        status_color = (0, 255, 0)
    else:
        drawing = False
        status_text = "Drawing: OFF"
        status_color = (0, 0, 255)
    
    # Update previous position
    prev_x, prev_y = hand_x, hand_y
    
    # Combine frame and canvas
    frame = cv2.bitwise_or(frame, canvas)
    
    # Add status information
    cv2.putText(frame, status_text, (10, height - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Position: ({hand_x}, {hand_y})", (10, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(frame, "Press 'C' to clear, 'ESC' to exit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Green circle = 'Hand' position", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show result
    cv2.imshow("Demo Hand Drawing", frame)
    
    # Handle keyboard input
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('c'):  # C key to clear canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        print("Canvas cleared!")

# Cleanup
cv2.destroyAllWindows()
print("Demo ended. Thanks for trying!")
