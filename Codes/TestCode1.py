import cv2

cap = cv2.VideoCapture(1)

# ROI definition
ret, frame = cap.read()
H, W, _ = frame.shape
roi_width = W // 3
roi_x1 = (W - roi_width) // 2
roi_x2 = roi_x1 + roi_width
roi_center = ((roi_x1 + roi_x2)//2, H//2)

# Person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Tracker setup
tracker = None
tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        # Search mode
        rects, _ = hog.detectMultiScale(frame, winStride=(4,4),
                                        padding=(8,8), scale=1.05)

        for (x, y, w, h) in rects:
            person_center = (x + w//2, y + h//2)

            # only if outside ROI
            if not (roi_x1 <= person_center[0] <= roi_x2):
                # Lock onto this person
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                tracking = True
                break

    else:
        # Track mode
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            person_center = (x + w//2, y + h//2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            error_x = person_center[0] - roi_center[0]
            error_y = person_center[1] - roi_center[1]
            print("Error:", (error_x, error_y))

    # Draw ROI
    cv2.rectangle(frame, (roi_x1, 0), (roi_x2, H), (255,0,0), 2)

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # quit
        break
    elif key == ord("c"):  # change target
        tracker = None
        tracking = False

cap.release()
cv2.destroyAllWindows()
