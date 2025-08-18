import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(1)

# ROI definition
ret, frame = cap.read()
H, W, _ = frame.shape
roi_width = W // 3
roi_x1 = (W - roi_width) // 2
roi_x2 = roi_x1 + roi_width
roi_center = ((roi_x1 + roi_x2)//2, H//2)

# Load YOLOv5 Nano
model = YOLO("yolov5n.pt")  # pretrained COCO model (includes "person")

# Tracker setup
tracker = None
tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        # Run YOLOv5n detection (low conf threshold for safety)
        results = model(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 0 = "person"
                if cls_id == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    person_center = (x1 + w//2, y1 + h//2)

                    # only if outside ROI
                    if not (roi_x1 <= person_center[0] <= roi_x2):
                        # Lock onto this person
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        tracking = True
                        break
            if tracking:  # break outer loop if person locked
                break

    else:
        # Track mode
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            person_center = (x + w//2, y + h//2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.line(frame, roi_center, person_center, (255,255,255))
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
