import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Create canvas
ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)
else:
    canvas = None

# Previous position
prev_x, prev_y = 0, 0
drawing = False

# Color range for skin detection (HSV)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

while True:
    success, frame = cap.read()
    if not success:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for skin color detection
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assumed to be hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 1000:  # Minimum area threshold
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Find the top point of the hand (assumed to be index finger tip)
            # This is a simplified approach - the highest point in the contour
            contour_points = largest_contour.reshape(-1, 2)
            top_point = tuple(contour_points[contour_points[:, 1].argmin()])
            
            # Draw circle at detected finger tip
            cv2.circle(frame, top_point, 10, (0, 255, 0), -1)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Simple drawing logic: draw when hand is in upper half of frame
            frame_height = frame.shape[0]
            if top_point[1] < frame_height // 2:
                # Hand is in upper half - enable drawing
                if drawing and prev_x != 0 and prev_y != 0:
                    # Draw line from previous position to current position
                    cv2.line(canvas, (prev_x, prev_y), top_point, (255, 0, 0), 5)
                drawing = True
            else:
                # Hand is in lower half - disable drawing
                drawing = False
            
            # Update previous position
            prev_x, prev_y = top_point
        else:
            # No significant contour found
            drawing = False
            prev_x, prev_y = 0, 0
    else:
        # No contours found
        drawing = False
        prev_x, prev_y = 0, 0
    
    # Combine frame and canvas
    if canvas is not None:
        frame = cv2.bitwise_or(frame, canvas)
    
    # Display instructions
    cv2.putText(frame, "Press 'C' to clear, 'ESC' to exit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Move hand to upper half to draw", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Lower half to stop drawing", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show mask in small window for debugging
    mask_resized = cv2.resize(mask, (200, 150))
    cv2.imshow("Mask", mask_resized)
    
    # Show result
    cv2.imshow("Simple Hand Drawing", frame)
    
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
