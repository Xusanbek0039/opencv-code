import cv2

print("Checking available cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i}: Available - {frame.shape}")
        else:
            print(f"Camera {i}: Available but can't read frame")
        cap.release()
    else:
        print(f"Camera {i}: Not available")
