import cv2
import time
import re
import socket
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import easyocr
import logging

HOST = '127.0.0.1'
PORT = 12346
GATE_COOLDOWN = 5

logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('ultralytics.engine.trainer').setLevel(logging.WARNING)
logging.getLogger('ultralytics.engine.predictor').setLevel(logging.WARNING)
logging.getLogger('ultralytics.engine.model').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
plate_model = YOLO("license_plate_detector.pt", verbose=False)
reader = easyocr.Reader(['en'], gpu=True)

cred = credentials.Certificate("psio-parking-firebase-adminsdk-gl8z1-55d95c00aa.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def is_valid_license_plate(plate_text):
    pattern = r'^[A-Z0-9]{6,8}$'
    return bool(re.match(pattern, plate_text))


def check_plate_in_database(plate_text):
    plates_ref = db.collection("parking_logs")
    return bool(plates_ref.where("license_plate", "==", plate_text).get())


def send_plate_to_server(plate_text):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall("ENTRY_CAMERA".encode())
        time.sleep(0.1)
        s.sendall(plate_text.encode())
        response = s.recv(1024).decode()
        print(f"Server Response: {response}")


def process_entry_camera():
    cap = cv2.VideoCapture(0)
    last_plate_text = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        plate_text = None
        plate_results = plate_model(frame)
        for plate_result in plate_results:
            for box in plate_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                result = reader.readtext(plate_img)
                if result:
                    plate_text = result[0][1].replace(" ", "").strip()

        if plate_text and is_valid_license_plate(plate_text):
            if plate_text != last_plate_text:
                if check_plate_in_database(plate_text):
                    print(f"Plate {plate_text} is valid. Sending to server...")
                    send_plate_to_server(plate_text)
                    last_plate_text = plate_text
                else:
                    print(f"Plate {plate_text} is not authorized.")

        cv2.imshow("Entry Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_entry_camera()
