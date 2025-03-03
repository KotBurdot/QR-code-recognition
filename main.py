import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import time

# для задержки рамки
last_detection_time = 0 #время последнего определения
last_detected_pts = None #координаты углов последнего обнаруженного qr

center_coordinates = None #координаты центра qr


def detect_and_draw_qr(frame, target_data):
    global last_detection_time, last_detected_pts, center_coordinates
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # декод qr кодов
    decoded_objects = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])

    min_distance = None
    current_time = time.time()
    detected = False

    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        if data == target_data:
            points = obj.polygon
            if len(points) == 4:
                pts = np.array(points, dtype=np.int32)
                last_detected_pts = pts
                last_detection_time = current_time
                detected = True

                # Вычисление центра qr
                center_x = int(sum(pt[0] for pt in pts) / 4)
                center_y = int(sum(pt[1] for pt in pts) / 4)
                center_coordinates = (center_x, center_y)

                # вычисление расстояния
                width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
                focal_length = 600  # для улучшенной точности
                real_qr_width = 8.5  # реальная ширина qr в см
                distance = (real_qr_width * focal_length) / width
                min_distance = distance if min_distance is None else min(min_distance, distance)

    # сохранение обводки 100 мс после исчезновения qr для плавности
    if not detected and last_detected_pts is not None and (current_time - last_detection_time) < 0.1:
        cv2.polylines(frame, [last_detected_pts], isClosed=True, color=(0, 255, 0), thickness=3)
        if center_coordinates:
            cv2.circle(frame, center_coordinates, 5, (0, 255, 0), -1)
    elif detected:
        cv2.polylines(frame, [last_detected_pts], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.circle(frame, center_coordinates, 5, (0, 255, 0), -1)

    if min_distance: #пишем расстояние до qr
        cv2.putText(frame, f"Distance: {min_distance:.2f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if center_coordinates: #пишем пишем координаты центра qr
        cv2.putText(frame, f"Position: {center_coordinates}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Ширина кадра
cap.set(4, 1080)  # Высота кадра
cap.set(cv2.CAP_PROP_FPS, 60)  # частота кадров

target_qr = "1"  # Целевое значение qr

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_draw_qr(frame, target_qr)
    print(center_coordinates)

    cv2.imshow('Decoder', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()