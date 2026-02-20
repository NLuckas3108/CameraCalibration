import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R_scipy
from getRealsenseIntrinsics import get_intrinsics

PATTERN_SIZE = (6, 7) #
SQUARE_SIZE = 0.030

objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

all_corners = []
all_obj_points = []
valid_indices = []
image_size = None

print("Starte Ecken-Detektion...")
for i in range(50):
    img_path = f"calibration_data/cam_{i:03d}.png"
    pose_path = f"calibration_data/pose_{i:03d}.npy"
    
    if not os.path.exists(img_path) or not os.path.exists(pose_path):
        print(f"Warnung: Paar {i:03d} fehlt. Wird übersprungen.")
        continue
        
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if image_size is None:
        image_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

    if ret:
        all_corners.append(corners)
        all_obj_points.append(objp)
        valid_indices.append(i)
        print(f"Bild {i:03d}: Ecken gefunden!")
    else:
        print(f"Bild {i:03d}: KEINE Ecken gefunden.")

print(f"\nVerwende {len(valid_indices)} von 50 Bildern für die Kalibrierung.\n")

#print("Berechne intrinsische Parameter...")
#ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
#    all_obj_points, all_corners, image_size, None, None
#)

#camera_matrix, dist_coeffs = get_intrinsics(width=640, height=480)

camera_matrix = np.array([
    [608.28442383, 0.,           322.55404663],
    [0.,           608.72912598, 242.2066803 ],
    [0.,           0.,           1.          ]
], dtype=np.float64)

dist_coeffs = np.zeros((5, 1), dtype=np.float64)

R_gripper2base_list = []
t_gripper2base_list = []
R_target2cam_list = []
t_target2cam_list = []

print("Berechne Board-Posen und bereite Roboter-Matrizen vor...")
for idx, objpoints, corners in zip(valid_indices, all_obj_points, all_corners):
    success, rvec, tvec = cv2.solvePnP(objpoints, corners, camera_matrix, dist_coeffs)
    
    if success:
        R_target2cam, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_target2cam)
        t_target2cam_list.append(tvec)
        
        pose_1d = np.load(f"calibration_data/pose_{idx:03d}.npy")
        
        t_base2tcp = (pose_1d[:3] / 1000.0).reshape(3, 1)
        
        r_matrix = R_scipy.from_euler('ZYZ', pose_1d[3:], degrees=True).as_matrix()
        R_base2tcp = r_matrix
        
        R_tcp2base = R_base2tcp.T
        t_tcp2base = -R_tcp2base @ t_base2tcp
        
        R_gripper2base_list.append(R_tcp2base)
        t_gripper2base_list.append(t_tcp2base)
        
print("Führe Hand-Auge-Kalibrierung aus...")
R_cam2base, t_cam2base = cv2.calibrateHandEye(
    R_gripper2base_list, t_gripper2base_list,
    R_target2cam_list, t_target2cam_list,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_cam2base = np.eye(4)
T_cam2base[:3, :3] = R_cam2base
T_cam2base[:3, 3] = t_cam2base.flatten()
T_base2cam = np.linalg.inv(T_cam2base)

print("\n--- ERGEBNIS ---")
print("Transformationsmatrix T_Cam_to_Base:")
print(T_cam2base)
print("Transformationsmatrix T_Base_to_cam:")
print(T_base2cam)

np.save("T_cam2base.npy", T_cam2base)
np.save("T_base2cam.npy", T_base2cam)