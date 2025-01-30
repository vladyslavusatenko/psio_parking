import threading

import cv2
import time
import re
import socket
from ultralytics import YOLO
import easyocr
from _datetime import datetime
import logging
import uuid


CAMERA_INDEX = 2
RESOLUTION = (640, 360)
PARKING_SPOTS = [
    (300, 60, 457, 132, 1),
    (300, 136, 457, 217, 2),
    (300, 222, 457, 300, 3),
    (300, 309, 460, 385, 4),
    (300, 394, 460, 464, 5),
    (577, 62, 734, 126, 6),
    (577, 134, 734, 213, 7),
    (577, 222, 734, 300, 8),
    (577, 307, 734, 386, 9),
    (577, 392, 734, 463, 10)
]
EXIT_AREA = (300, 490, 412, 682)

PARKING_THRESHOLD = 10
OVERLAP_THRESHOLD = 0.35
VIOLATION_TIME_THRESHOLD = 7
MOVEMENT_THRESHOLD = 5
TRACK_TIMEOUT = 5


HOST = '127.0.0.1'
PORT = 12346


logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('ultralytics.engine.trainer').setLevel(logging.WARNING)
logging.getLogger('ultralytics.engine.predictor').setLevel(logging.WARNING)
logging.getLogger('ultralytics.engine.model').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


car_model = YOLO("best_car_detection_812.pt", verbose=False)
plate_model = YOLO("license_plate_detector.pt", verbose=False)
reader = easyocr.Reader(['en'], gpu=True)


tracked_cars = {}
car_positions = {}
violation_tracker = {}
violation_notified = set()
car_plates = {}
exited_cars = set()


class TrackedCar:
    def __init__(self, x1, y1, x2, y2, plate_text=None):
        self.id = uuid.uuid4()
        self.positions = [(x1, y1, x2, y2)]
        self.last_seen = time.time()
        self.plate_text = plate_text
        self.in_exit_area = False

    def update_position(self, x1, y1, x2, y2):
        self.positions.append((x1, y1, x2, y2))
        self.last_seen = time.time()


def is_valid_license_plate(plate_text):
    pattern = r'^[A-Z0-9]{6,8}$'
    return bool(re.match(pattern, plate_text))


def detect_license_plate(frame, car_bbox):
    x1, y1, x2, y2 = car_bbox
    car_img = frame[y1:y2, x1:x2]

    try:
        plate_results = plate_model(car_img)
        for plate_result in plate_results:
            for box in plate_result.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                plate_img = car_img[py1:py2, px1:px2]
                result = reader.readtext(plate_img)
                if result:
                    plate_text = result[0][1].replace(" ", "").strip()
                    if is_valid_license_plate(plate_text):
                        return plate_text
    except Exception as e:
        print(f"Error detecting license plate: {e}")
    return None


def detect_cars(frame):
    cars = []
    results = car_model(frame)

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = result.names[int(cls)]
            if 'car' in label.lower():
                plate_text = detect_license_plate(frame, (x1, y1, x2, y2))
                cars.append((x1, y1, x2, y2, conf, plate_text))
    return cars


def is_car_moving(car_id, current_pos):
    if car_id not in car_positions:
        car_positions[car_id] = (current_pos, time.time())
        return True

    old_pos, _ = car_positions[car_id]
    old_x1, old_y1, old_x2, old_y2 = old_pos
    x1, y1, x2, y2 = current_pos

    old_center_x = (old_x1 + old_x2) // 2
    old_center_y = (old_y1 + old_y2) // 2
    new_center_x = (x1 + x2) // 2
    new_center_y = (y1 + y2) // 2

    distance = ((new_center_x - old_center_x) ** 2 + (new_center_y - old_center_y) ** 2) ** 0.5
    return distance > MOVEMENT_THRESHOLD


def calculate_overlap(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x1b, y1b, x2b, y2b = rect2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_rect1 = (x2 - x1) * (y2 - y1)
        area_rect2 = (x2b - x1b) * (y2b - y1b)
        reference_area = min(area_rect1, area_rect2)
        return intersection_area / reference_area
    return 0


def notify_server_violation(violation_type, car_id, plate_text=None, violation_time=None):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))

            s.sendall("PARKING_VIOLATION".encode())
            time.sleep(0.1)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if violation_time is None else violation_time
            violation_data = f"{car_id},{plate_text or 'UNKNOWN'},{current_time}, {violation_type}"

            s.sendall(violation_data.encode())

            response = s.recv(1024).decode()
            print(f"Server response: {response}")

    except Exception as e:
        print(f"Error sending violation: {e}")

def send_exit_data(tracked_car_id):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((HOST, PORT))

                s.sendall("CAR_EXITED".encode())
                time.sleep(0.1)

                exit_data = f"{tracked_car_id}"
                s.sendall(exit_data.encode())

                response = s.recv(1024).decode()
                print(f"Server response: {response}")

        except Exception as e:
            print(f"Error sending exit data: {e}")


def notify_server_exit(tracked_car):
    thread = threading.Thread(target=send_exit_data, args=(tracked_car.id,))
    thread.start()


def check_parking_status(car, tracked_car):
    x1, y1, x2, y2, conf, plate_text = car
    current_time = time.time()
    overlapping_spots = []
    overlap_values = []

    is_moving = is_car_moving(tracked_car.id, (x1, y1, x2, y2))
    car_positions[tracked_car.id] = ((x1, y1, x2, y2), current_time)

    if plate_text and not tracked_car.plate_text:
        tracked_car.plate_text = plate_text

    for spot in PARKING_SPOTS:
        spot_x1, spot_y1, spot_x2, spot_y2, spot_id = spot
        overlap = calculate_overlap((x1, y1, x2, y2), (spot_x1, spot_y1, spot_x2, spot_y2))
        if overlap > OVERLAP_THRESHOLD:
            overlapping_spots.append(spot_id)
            overlap_values.append(overlap)

    exit_overlap = calculate_overlap((x1, y1, x2, y2), EXIT_AREA)
    if exit_overlap > OVERLAP_THRESHOLD:
        tracked_car.in_exit_area = True
        return "exit_area", None
    else:
        tracked_car.in_exit_area = False

    if is_moving:
        if tracked_car.id in violation_tracker:
            del violation_tracker[tracked_car.id]
        if tracked_car.id in violation_notified:
            violation_notified.remove(tracked_car.id)

    if len(overlapping_spots) == 1 and overlap_values[0] > 0.6:
        if tracked_car.id in violation_tracker:
            del violation_tracker[tracked_car.id]
        return "correct", overlapping_spots[0]

    if len(overlapping_spots) > 1:
        significant_overlaps = sum(1 for overlap in overlap_values if overlap > 0.3)
        if significant_overlaps > 1:
            if tracked_car.id not in violation_tracker:
                violation_tracker[tracked_car.id] = current_time
            elif current_time - violation_tracker[tracked_car.id] > VIOLATION_TIME_THRESHOLD:
                if tracked_car.id not in violation_notified:
                    plate = tracked_car.plate_text
                    notify_server_violation("wrong_parking", tracked_car.id, plate)
                    violation_notified.add(tracked_car.id)
                return "wrong_parking", None
            return "potential_violation", None

    if len(overlapping_spots) == 0 or max(overlap_values, default=0) < OVERLAP_THRESHOLD:
        if tracked_car.id not in violation_tracker:
            violation_tracker[tracked_car.id] = current_time
        elif current_time - violation_tracker[tracked_car.id] > VIOLATION_TIME_THRESHOLD:
            if tracked_car.id not in violation_notified:
                plate = tracked_car.plate_text
                notify_server_violation("blocked_way", tracked_car.id, plate)
                violation_notified.add(tracked_car.id)
            return "blocked_way", None
        return "potential_violation", None

    return "monitoring", None


def get_occupied_spots(cars):
    occupied = {}
    for tracked_car_id, tracked_car in tracked_cars.items():
        for car in cars:
            x1, y1, x2, y2, conf, plate_text = car
            overlap = calculate_overlap((x1, y1, x2, y2), (
            tracked_car.positions[-1][0], tracked_car.positions[-1][1], tracked_car.positions[-1][2],
            tracked_car.positions[-1][3]))
            if overlap > 0.5:
                status, spot_id = check_parking_status(car, tracked_car)
                if status == "correct":
                    occupied[spot_id] = tracked_car.id
    return occupied


def update_tracked_cars(cars):
    global tracked_cars
    new_tracked_cars = {}
    assigned_car_ids = set()
    for car in cars:
        x1, y1, x2, y2, conf, plate_text = car
        assigned = False
        for tracked_car_id, tracked_car in tracked_cars.items():
            overlap = calculate_overlap((x1, y1, x2, y2), (
            tracked_car.positions[-1][0], tracked_car.positions[-1][1], tracked_car.positions[-1][2],
            tracked_car.positions[-1][3]))
            if overlap > 0.5:
                tracked_car.update_position(x1, y1, x2, y2)
                if plate_text and not tracked_car.plate_text:
                    tracked_car.plate_text = plate_text
                new_tracked_cars[tracked_car_id] = tracked_car
                assigned_car_ids.add(tracked_car_id)
                assigned = True
                break

        if not assigned:
            new_car = TrackedCar(x1, y1, x2, y2, plate_text)
            new_tracked_cars[new_car.id] = new_car

    tracked_cars_to_remove = []
    for tracked_car_id, tracked_car in tracked_cars.items():
        if tracked_car_id not in assigned_car_ids:
            if time.time() - tracked_car.last_seen > TRACK_TIMEOUT:
                tracked_cars_to_remove.append(tracked_car_id)
            else:
                new_tracked_cars[tracked_car_id] = tracked_car
    for tracked_car_id in tracked_cars_to_remove:
        del tracked_cars[tracked_car_id]

    tracked_cars = new_tracked_cars
    return cars


def draw_objects(frame, cars):
    current_time = time.time()
    occupied_spots = get_occupied_spots(cars)

    cv2.rectangle(frame, (EXIT_AREA[0], EXIT_AREA[1]),
                  (EXIT_AREA[2], EXIT_AREA[3]), (255, 165, 0), 2)
    cv2.putText(frame, "Exit Area", (EXIT_AREA[0], EXIT_AREA[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    for spot in PARKING_SPOTS:
        spot_x1, spot_y1, spot_x2, spot_y2, spot_id = spot
        if any(spot_id == key for key in occupied_spots):
            color = (0, 255, 0)
            status = "Occupied"
        else:
            color = (255, 0, 0)
            status = "Empty"

        cv2.rectangle(frame, (spot_x1, spot_y1), (spot_x2, spot_y2), color, 2)
        cv2.putText(frame, f"Spot {spot_id}: {status}",
                    (spot_x1, spot_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    for tracked_car_id, tracked_car in tracked_cars.items():
        for car in cars:
            x1, y1, x2, y2, conf, plate_text = car
            overlap = calculate_overlap((x1, y1, x2, y2), (
            tracked_car.positions[-1][0], tracked_car.positions[-1][1], tracked_car.positions[-1][2],
            tracked_car.positions[-1][3]))
            if overlap > 0.5:
                status, spot_id = check_parking_status(car, tracked_car)

                if status == "correct":
                    color = (0, 255, 0)
                    text = f"Parked in {spot_id}"
                elif status == "wrong_parking":
                    color = (0, 0, 255)
                    text = "Wrong Parking!"
                elif status == "blocked_way":
                    color = (0, 0, 255)
                    text = "Blocked Way!"
                elif status == "potential_violation":
                    color = (0, 165, 255)
                    remaining_time = VIOLATION_TIME_THRESHOLD - (
                            current_time - violation_tracker.get(tracked_car.id, current_time))
                    text = f"Potential Violation ({int(remaining_time)}s)"
                elif status == "exit_area":
                    color = (255, 165, 0)
                    text = "In Exit Area"
                else:
                    color = (255, 255, 0)
                    text = "Monitoring"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if plate_text:
                    cv2.putText(frame, f"Plate: {plate_text}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def monitor_parking():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Nie można otworzyć kamery parkingowej.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cars = detect_cars(frame)
            cars = update_tracked_cars(cars)

            for tracked_car_id, tracked_car in tracked_cars.items():
                for car in cars:
                    x1, y1, x2, y2, conf, plate_text = car
                    overlap = calculate_overlap((x1, y1, x2, y2), (
                    tracked_car.positions[-1][0], tracked_car.positions[-1][1], tracked_car.positions[-1][2],
                    tracked_car.positions[-1][3]))
                    if overlap > 0.5:
                        status, spot_id = check_parking_status(car, tracked_car)
                        if status == "exit_area" and tracked_car.id not in exited_cars:
                            notify_server_exit(tracked_car)
                            exited_cars.add(tracked_car.id)
            draw_objects(frame, cars)
            cv2.imshow("Parking Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor_parking()