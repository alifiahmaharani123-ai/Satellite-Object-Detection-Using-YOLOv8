# webcam_detect.py
from ultralytics import YOLO
import cv2
import time

# Pakai pretrained YOLOv8s (atau ganti 'best.pt' kalau punya)
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("Gagal membuka kamera")
    exit(1)

print("Tekan 's' untuk START/STOP deteksi. Tekan 'q' untuk keluar.")

running = False
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if running:
        # Gunakan stream=True untuk performa lebih baik (mengembalikan generator)
        # tapi di sini kita gunakan model.predict(frame) per-frame
        results = model.predict(frame, imgsz=640, conf=0.45, verbose=False)
        # Ambil hasil ter-annotate
        annotated = results[0].plot()
        display_frame = annotated

    cv2.imshow("YOLOv8 Live (Press 's' to toggle, 'q' to quit)", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        running = not running
        print("Deteksi:", "ON" if running else "OFF")
        time.sleep(0.2)  # debouncing
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
