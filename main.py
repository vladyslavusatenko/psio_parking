import cv2
# import pytesseract
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from collections import deque
import time
import re
import easyocr

GATE_COOLDOWN = 5
plate_last_opened = {}

cred = credentials.Certificate("psio-parking-firebase-adminsdk-gl8z1-55d95c00aa.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


plate_model = YOLO("license_plate_detector.pt")
# car_model = YOLO("yolov8n.pt")


reader = easyocr.Reader(['en'],  gpu=True)


ENTRY_ZONE = (0, 25, 958, 535)
# EXIT_ZONE = (486, 25, 958, 535)

recent_plates = deque(maxlen=5)
last_logged_time = {}

def check_plate_in_database(plate_text):
    return True

def open_gate(gate_type):
    print(f"Opening {gate_type} gate.")


def close_gate(gate_type):
    print(f"Closing {gate_type} gate.")


def check_plate_in_database(plate_text):
    plates_ref = db.collection("parking_logs")
    existing_paltes = plates_ref.where("license_plate", "==", plate_text).get()
    return bool(existing_paltes)
def log_to_firebase(data):
    plates_ref = db.collection("parking_logs")
    existing_plates = plates_ref.where("license_plate", "==", data["license_plate"]).get()

    if not existing_plates:
        db.collection("parking_logs").add(data)
        print("Logged to Firebase:", data)
    else:
        print(f"Plate {data['license_plate']} already exists in the database. Skipping log.")

# def is_valid_license_plate(plate_text):
#     pattern = r'^[A-Z0-9]{6,8}$'  # Plates with 6-8 alphanumeric characters (you can modify this pattern)
#     return bool(re.match(pattern, plate_text))
#


# def should_log_plate(plate_text):
#     current_time = time.time()
#     #
#     # if not is_valid_license_plate(plate_text):
#     #     return False  # Reject invalid plates
#     if plate_text in last_logged_time:
#         time_since_last_log = current_time - last_logged_time[plate_text]
#         if time_since_last_log < 5:
#             return False
#
#     recent_plates.append(plate_text)
#     last_logged_time[plate_text] = current_time
#     return True

def detect_vehicles_and_plates(frame):
    plate_results = plate_model(frame, save=False, verbose=False)
    # car_results = car_model(frame, save=False, verbose=False)

    gate_type = None
    plate_text = None

    for plate_result in plate_results:
        for box in plate_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = plate_model.names[cls] if cls < len(plate_model.names) else "Unknown"

            if label == "license_plate":
                if x1 >= ENTRY_ZONE[0] and x2 <= ENTRY_ZONE[2] and y1 >= ENTRY_ZONE[1] and y2 <= ENTRY_ZONE[3]:
                    gate_type = "entry"
                # elif x1 >= EXIT_ZONE[0] and x2 <= EXIT_ZONE[2] and y1 >= EXIT_ZONE[1] and y2 <= EXIT_ZONE[3]:
                #     gate_type = "exit"


                plate_img = frame[y1:y2, x1:x2]
                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                _, thresh_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)

                result = reader.readtext(thresh_plate)
                if result:
                    plate_text = result[0][1].strip().replace(' ', '')

                color = (0, 255, 0) if gate_type == "entry" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Process vehicle detections
    # for car_result in car_results:
    #     for box in car_result.boxes:
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         cls = int(box.cls[0])  # Class index
    #         label = car_model.names[cls] if cls < len(car_model.names) else "Unknown"
    #
    #         if label == "car":  # Adjust based on your model's labels
    #             # Draw bounding box for vehicles
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #             cv2.putText(frame, f"{label}", (x1, y1 - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return plate_text, gate_type


def process_single_camera():
    cap = cv2.VideoCapture(0)

    desired_width = 640
    desired_height = 360

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame_resized = cv2.resize(frame, (desired_width, desired_height))

        plate_text, gate_type = detect_vehicles_and_plates(frame_resized)

        if plate_text:
            current_time = time.time()

            if check_plate_in_database(plate_text) and gate_type == "entry":
                if plate_text not in plate_last_opened or (
                        current_time - plate_last_opened[plate_text]) > GATE_COOLDOWN:
                    open_gate(gate_type)
                    plate_last_opened[plate_text] = current_time

                    time.sleep(GATE_COOLDOWN)
                    close_gate(gate_type)

        frame_output = cv2.resize(frame_resized, (frame.shape[1], frame.shape[0]))

        cv2.rectangle(frame_output, (ENTRY_ZONE[0], ENTRY_ZONE[1]), (ENTRY_ZONE[2], ENTRY_ZONE[3]), (0, 255, 0), 2)
        cv2.putText(frame_output, "ENTRY ZONE", (ENTRY_ZONE[0], ENTRY_ZONE[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.rectangle(frame_output, (EXIT_ZONE[0], EXIT_ZONE[1]), (EXIT_ZONE[2], EXIT_ZONE[3]), (0, 0, 255), 2)
        # cv2.putText(frame_output, "EXIT ZONE", (EXIT_ZONE[0], EXIT_ZONE[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Parking System", frame_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_single_camera()

