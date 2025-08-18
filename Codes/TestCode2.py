import cv2

cap = cv2.VideoCapture(1)

# ROI definition
ret, frame = cap.read()
H, W, _ = frame.shape
roi_width = W // 3
roi_x1 = (W - roi_width) // 2
roi_x2 = roi_x1 + roi_width
roi_center = ((roi_x1 + roi_x2)//2, H//2)

# Load MobileNet SSD model
prototxt = "deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# COCO/VOC classes for MobileNetSSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Tracker setup
tracker = None
tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not tracking:
        # Prepare input blob for SSD
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:  # threshold
                idx = int(detections[0, 0, i, 1])

                # Only detect "person"
                if CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * [W, H, W, H]
                    (x1, y1, x2, y2) = box.astype("int")
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
