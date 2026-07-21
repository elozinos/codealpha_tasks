import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def main():
    # 1. Initialize YOLOv8 model explicitly on CPU
    # Passing task='detect' fixes the "Unable to guess model task" warning
    model = YOLO("best3.pt", task="detect")

    # Access COCO class names safely directly from the underlying model architecture
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else model.names

    # 2. Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)

    # 3. Open webcam
    cap = cv2.VideoCapture("vid.mp4")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("CPU-Optimized Tracking started. Press 'q' inside the window to quit.")

    frame_count = 0
    tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run heavy YOLO + DeepSORT math every 2nd frame to save CPU workload
        if frame_count % 2 == 0:
            # imgsz=320 makes CPU detection significantly faster
            results = model(frame, verbose=False, imgsz=320)[0]

            detections = []
            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                left = int(xyxy[0])
                top = int(xyxy[1])
                width = int(xyxy[2] - xyxy[0])
                height = int(xyxy[3] - xyxy[1])

                detections.append(([left, top, width, height], conf, cls))

            # Update active tracking objects
            tracks = tracker.update_tracks(detections, frame=frame)

        # Draw the active tracks on EVERY frame for smooth visual output
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            class_id = track.get_det_class()

            # Retrieve class name using the safe dictionary lookup
            class_name = names.get(class_id, "Object") if isinstance(names, dict) else "Object"

            # Green box + ID label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display window
        cv2.imshow('YOLOv8 + DeepSORT (CPU Mode)', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program closed clean.")


if __name__ == "__main__":
    main()