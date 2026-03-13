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

    # 1. PnP für alle gültigen Bilder berechnen
    R_target2cam_list = []
    t_target2cam_list = []
    T_target2cam_list_4x4 = []
    synced_indices = []

    for idx, objpoints, corners in zip(valid_indices, all_obj_points, all_corners):
        success, rvec, tvec = cv2.solvePnP(objpoints, corners, camera_matrix, dist_coeffs)
        if success:
            R_target2cam, _ = cv2.Rodrigues(rvec)
            R_target2cam_list.append(R_target2cam)
            t_target2cam_list.append(tvec)
            
            T = np.eye(4)
            T[:3, :3] = R_target2cam
            T[:3, 3] = tvec.flatten()
            T_target2cam_list_4x4.append(T)
            synced_indices.append(idx)

    # 2. Setup für den Auto-Solver
    best_std = float('inf')
    best_config = None
    best_T_cam2base = None
    
    euler_conventions = ['zyz', 'ZYZ', 'zyx', 'ZYX', 'xyz', 'XYZ']
    inversion_modes = [True, False]
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF
    }

    print("Führe Auto-Solver aus, um die korrekte Matrizen-Konvention zu finden...")

    # Grid-Search über alle mathematischen Möglichkeiten
    for euler in euler_conventions:
        for invert in inversion_modes:
            for method_name, method_flag in methods.items():
                
                R_r_list = []
                t_r_list = []
                T_tcp2base_list = []
                valid_run = True
                
                for idx in synced_indices:
                    pose_path = os.path.join(cam_folder, f"pose_{idx:03d}.npy")
                    pose_1d = np.load(pose_path)
                    
                    t_tcp2base = (pose_1d[:3] / 1000.0).reshape(3, 1)
                    try:
                        R_tcp2base = R_scipy.from_euler(euler, pose_1d[3:], degrees=True).as_matrix()
                    except ValueError:
                        valid_run = False
                        break
                        
                    T_tcp2base = np.eye(4)
                    T_tcp2base[:3, :3] = R_tcp2base
                    T_tcp2base[:3, 3] = t_tcp2base.flatten()
                    T_tcp2base_list.append(T_tcp2base)
                    
                    if invert:
                        T_base2tcp = np.linalg.inv(T_tcp2base)
                        R_r_list.append(T_base2tcp[:3, :3])
                        t_r_list.append(T_base2tcp[:3, 3].reshape(3, 1))
                    else:
                        R_r_list.append(T_tcp2base[:3, :3])
                        t_r_list.append(T_tcp2base[:3, 3].reshape(3, 1))
                
                if not valid_run:
                    continue
                
                # Hand-Eye Berechnung versuchen
                try:
                    R_cam2base_est, t_cam2base_est = cv2.calibrateHandEye(
                        R_r_list, t_r_list,
                        R_target2cam_list, t_target2cam_list,
                        method=method_flag
                    )
                except Exception:
                    continue
                    
                if R_cam2base_est is None:
                    continue
                    
                T_cam2base = np.eye(4)
                T_cam2base[:3, :3] = R_cam2base_est
                T_cam2base[:3, 3] = t_cam2base_est.flatten()
                
                translations_target2tcp = []
                for i in range(len(synced_indices)):
                    T_t2c = T_target2cam_list_4x4[i]
                    T_tcp2b = T_tcp2base_list[i]
                    
                    # T_target^tcp = (T_tcp^base)^-1 * T_cam^base * T_target^cam
                    T_target2tcp = np.linalg.inv(T_tcp2b) @ T_cam2base @ T_t2c
                    translations_target2tcp.append(T_target2tcp[:3, 3])
                    
                translations_target2tcp = np.array(translations_target2tcp)
                
                # Abweichung in mm berechnen
                std_x = np.std(translations_target2tcp[:, 0])
                std_y = np.std(translations_target2tcp[:, 1])
                std_z = np.std(translations_target2tcp[:, 2])
                mean_std_mm = (std_x + std_y + std_z) / 3.0 * 1000.0 
                
                if mean_std_mm < best_std:
                    best_std = mean_std_mm
                    best_config = (euler, invert, method_name)
                    best_T_cam2base = T_cam2base

    if best_T_cam2base is None:
        print("Fehler: Kalibrierung fehlgeschlagen. Keine Kombination ergab ein Ergebnis.")
        return

    print(f"\nBeste Konfiguration gefunden:")
    print(f" - Euler-Konvention: {best_config[0]}")
    print(f" - Roboter-Pose invertiert: {best_config[1]}")
    print(f" - OpenCV Algorithmus: {best_config[2]}")
    print(f" - Durchschnittliche Abweichung (Jitter) des Musters: {best_std:.2f} mm")

    if best_std > 20.0:
        print("WARNUNG: Die Abweichung ist sehr hoch! Möglicherweise ist das Schachbrett-Muster")
        print("nicht 30mm groß oder die Bilder/Posen sind nicht synchronisiert.")

    T_base2cam = np.linalg.inv(best_T_cam2base)
    
    print(f"\n--- ERGEBNIS FÜR {cam_folder} ---")
    print("T_cam2base (Pose der Kamera im Roboter-Base-Frame):\n", best_T_cam2base)
    print("\nT_base2cam (Pose der Base im Kamera-Frame):\n", T_base2cam)

    np.save(os.path.join(cam_folder, "T_cam2base.npy"), best_T_cam2base)
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