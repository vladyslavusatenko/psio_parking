import socket
import threading
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import time
from datetime import datetime

HOST = '127.0.0.1'
PORT = 12346

cred = credentials.Certificate("psio-parking-firebase-adminsdk-gl8z1-55d95c00aa.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

vehicle_plate = None
vehicle_parked = False
plate_lock = threading.Lock()


def log_vehicle_event(plate_text, status):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    doc_name = f"{plate_text}_{status}_{timestamp}"

    log_ref = db.collection("entry_logs").document(doc_name)
    log_ref.set({
        "license_plate": plate_text,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "status": status
    })
    print(f"✅ Log saved as '{doc_name}' for vehicle {plate_text} ({status})")

def log_violation_event(vehicle_id, plate_number, violation_time, violation_type):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    doc_name = f"violation_{vehicle_id}_{timestamp}"

    try:
        log_ref = db.collection("violations").document(doc_name)
        log_ref.set({
            "vehicle_id": vehicle_id,
            "license_plate": plate_number,
            "violation_time": violation_time,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "violation_type": violation_type
        })
        print(f"✅ Violation log saved as '{doc_name}' for vehicle with plate {plate_number}")
    except Exception as e:
        print(f"Error saving violation log: {e}")
    return doc_name

def log_exit_event(vehicle_id, status):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    doc_name = f"exit_{vehicle_id}_{timestamp}"

    try:
        log_ref = db.collection("exit_logs").document(doc_name)
        log_ref.set({
            "vehicle_id": vehicle_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
            " status": status
        })
        print(f"✅ Exit log saved as '{doc_name}' for vehicle with id {vehicle_id}")
    except Exception as e:
        print(f"Error saving exit log: {e}")
    return doc_name

def open_entry_gate():
    print("🚗 Brama wjazdowa otwarta.")
    time.sleep(7)
    close_entry_gate()

def close_entry_gate():
    print("🚧 Brama wjazdowa zamknięta.")


def open_exit_gate():
    print("🚗 Brama wyjazdowa otwarta.")
    time.sleep(7)
    close_exit_gate()


def close_exit_gate():
    print("🚧 Brama wyjazdowa zamknięta.")


def validate_plate_in_db(plate_text):
    plates_ref = db.collection("parking_logs")
    plate_query = plates_ref.where(filter=FieldFilter("license_plate", "==", plate_text)).get()
    return bool(plate_query)


def handle_entry_camera(conn):
    global vehiqcle_plate
    try:
        data = conn.recv(1024)
        if data:
            plate_text = data.decode().strip()
            print(f"📸 Received Plate from Entry Camera: {plate_text}")

            is_plate_valid = validate_plate_in_db(plate_text)
            if is_plate_valid:
                with plate_lock:
                    vehicle_plate = plate_text
                conn.sendall("Plate is valid.".encode())
                log_vehicle_event(plate_text, "entry")
                open_entry_gate()
            else:
                conn.sendall("Plate is not valid.".encode())
                print("⛔ Alert: Unauthorized vehicle.")
    except Exception as e:
        print(f"Error handling entry camera data: {e}")
        conn.sendall("Error processing entry data".encode())
    finally:
        conn.close()

def handle_parking_violation(conn):
    try:
        data = conn.recv(1024)
        if data:
            violation_data = data.decode().strip()
            vehicle_id, plate_number, violation_time, violation_type = violation_data.split(',')

            try:
                formatted_time = datetime.strptime(violation_time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                formatted_time = datetime.now()

            print(f"⛔ Violation detected:")
            print(f"  Vehicle ID: {vehicle_id}")
            print(f"  Plate Number: {plate_number}")
            print(f"  Violation Time: {formatted_time}")
            print(f"  Violation Type: {violation_type}")

            res = log_violation_event(vehicle_id, plate_number, violation_time, violation_type)

            response = f"Violation for vehicle {vehicle_id} with license plate {plate_number} logged successfully as {res}"
            conn.sendall(response.encode())
    except Exception as e:
        print(f"Error handling violation data: {e}")
        conn.sendall("Error processing violation".encode())
    finally:
        conn.close()

def handle_exit_camera(conn):
    try:
        data = conn.recv(1024)
        if data:
            vehicle_id = data.decode().strip()
            print(f"🚗 Vehicle {vehicle_id} is exiting.")
            log_exit_event(vehicle_id, "exit")
            open_exit_gate()
            conn.sendall("Exit gate opened.".encode())
    except Exception as e:
        print(f"Error handling exit data: {e}")
        conn.sendall("Error processing exit data.".encode())
    finally:
        conn.close()


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f"✅ Server is listening on {HOST}:{PORT}")

        while True:
            try:
                conn, addr = server.accept()
                print(f"[SERVER] New connection from {addr}")

                data = conn.recv(1024)
                if not data:
                    print("[SERVER] No initial data received")
                    conn.close()
                    continue

                client_type = data.decode().strip()
                print(f"[SERVER] Client type: {client_type}")

                if client_type == "PARKING_VIOLATION":
                    handle_parking_violation(conn)
                elif client_type == "CAR_EXITED":
                    handle_exit_camera(conn)
                elif client_type == "ENTRY_CAMERA":
                   handle_entry_camera(conn)
                else:
                    print(f"[SERVER] Unknown client type: {client_type}")
                    conn.close()

            except Exception as e:
                print(f"[SERVER] Error in main loop: {e}")
                continue


if __name__ == "__main__":
    start_server()