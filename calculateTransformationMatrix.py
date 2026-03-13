import cv2
import numpy as np
import os
import sys
import argparse
from scipy.spatial.transform import Rotation as R_scipy
from getRealsenseIntrinsics import get_intrinsics

def analyze_camera_folder(cam_folder):
    PATTERN_SIZE = (6, 7)
    SQUARE_SIZE = 0.030

    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    all_corners = []
    all_obj_points = []
    valid_indices = []
    image_size = None

    print(f"\nStarte Analyse für {cam_folder}...")
    
    for i in range(50):
        img_path = os.path.join(cam_folder, f"cam_{i:03d}.png")
        pose_path = os.path.join(cam_folder, f"pose_{i:03d}.npy")
        
        if not os.path.exists(img_path) or not os.path.exists(pose_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

        if ret:
            all_corners.append(corners)
            all_obj_points.append(objp)
            valid_indices.append(i)

    print(f"Verwende {len(valid_indices)} Bilder für die Kalibrierung in {cam_folder}.")

    if len(valid_indices) < 3:
        print(f"Fehler: Nicht genug gültige Bilder in {cam_folder} gefunden. Breche für diese Kamera ab.")
        return
    
    serial_file = os.path.join(cam_folder, "serial.txt")
    if not os.path.exists(serial_file):
        print(f"Fehler: Keine serial.txt in {cam_folder} gefunden. Überspringe...")
        return
        
    with open(serial_file, "r") as f:
        serial_number = f.read().strip()
        
    print(f"Lade Intrinsics für Kamera mit Seriennummer: {serial_number}")
    camera_matrix, dist_coeffs = get_intrinsics(serial_number, width=640, height=480)

    R_base2tcp_list = []
    t_base2tcp_list = []
    R_target2cam_list = []
    t_target2cam_list = []

    for idx, objpoints, corners in zip(valid_indices, all_obj_points, all_corners):
        success, rvec, tvec = cv2.solvePnP(objpoints, corners, camera_matrix, dist_coeffs)
        
        if success:
            R_target2cam, _ = cv2.Rodrigues(rvec)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec)
            
            pose_path = os.path.join(cam_folder, f"pose_{idx:03d}.npy")
            pose_1d = np.load(pose_path)
            
            # 1. TCP-Pose im Base-System berechnen
            t_tcp2base = (pose_1d[:3] / 1000.0).reshape(3, 1)
            # WICHTIG: 'ZYZ' zwingend groß schreiben für intrinsische Rotation!
            R_tcp2base = R_scipy.from_euler('ZYZ', pose_1d[3:], degrees=True).as_matrix()
            
            T_tcp2base = np.eye(4)
            T_tcp2base[:3, :3] = R_tcp2base
            T_tcp2base[:3, 3] = t_tcp2base.flatten()
            
            # 2. Invertieren für Eye-to-Hand Kalibrierung
            # OpenCV erwartet die Pose der Base aus Sicht des TCPs
            T_base2tcp = np.linalg.inv(T_tcp2base)
            
            # Listen füttern (mit der invertierten Pose)
            R_base2tcp_list.append(T_base2tcp[:3, :3])
            t_base2tcp_list.append(T_base2tcp[:3, 3].reshape(3, 1))

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2tcp_list, t_base2tcp_list,
        R_target2cam_list, t_target2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()
    T_base2cam = np.linalg.inv(T_cam2base)

    print(f"--- ERGEBNIS FÜR {cam_folder} ---")
    print("T_Cam_to_Base:\n", T_cam2base)
    print("T_Base_to_Cam:\n", T_base2cam)

    np.save(os.path.join(cam_folder, "T_cam2base.npy"), T_cam2base)
    np.save(os.path.join(cam_folder, "T_base2cam.npy"), T_base2cam)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auswertung der Hand-Eye-Kalibrierung für mehrere Kameras")
    parser.add_argument("data_folder", type=str, help="Pfad zum Hauptordner (z.B. calibration_data_123456)")
    args = parser.parse_args()

    base_folder = args.data_folder
    if not os.path.exists(base_folder):
        print(f"Fehler: Der Ordner '{base_folder}' existiert nicht.")
        sys.exit(1)

    cam_folders = [f.path for f in os.scandir(base_folder) if f.is_dir() and f.name.startswith("camera_")]

    if not cam_folders:
        print(f"Keine Kamera-Unterordner in '{base_folder}' gefunden.")
        sys.exit(1)

    for cam_folder in sorted(cam_folders):
        analyze_camera_folder(cam_folder)
        
    print("\nAlle Kameras erfolgreich ausgewertet.")