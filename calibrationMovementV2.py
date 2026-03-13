import rclpy
import threading
import rclpy.executors
import os
import time
import cv2        
import numpy as np
import pyrealsense2 as rs
import traceback 

ROBOT_ID   = "dsr01"
ROBOT_MODEL= "m1013"

import DR_init
DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def extract_pose(raw_pose_data):
    """Extrahiert die 6 Werte aus der Doosan-Pose."""
    result = []
    for item in raw_pose_data:
        if isinstance(item, (list, tuple)):
            result.append(float(item[0]))
        else:
            result.append(float(item))
    return result[:6]


def main(args=None):
    movement_finished_event = threading.Event()
    
    rclpy.init(args=args)
    node = rclpy.create_node('dsr_calibration_node', namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    try:
        from DSR_ROBOT2 import movej, get_current_posx, DR_BASE
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        return
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    movement_thread = threading.Thread(
        target=run_movement,
        args=(movej, get_current_posx, DR_BASE, movement_finished_event)
    )
    movement_thread.start()

    while rclpy.ok() and not movement_finished_event.is_set():
        time.sleep(0.1)

    node.get_logger().info("Fahre Executor herunter...")
    executor.shutdown()
    executor_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    

def run_movement(movej, get_current_posx, DR_BASE, movement_finished_event):
    try:
        # HIER DIE AUSGABEN VOM LOGGER EINTRAGEN
        waypoints_cam1 = [
            [19.64, 33.05, 132.47, -3.43, 195.42, 154.59],
            [16.24, 44.94, 101.37, -0.64, 195.43, 154.60],
            [21.44, 46.16, 98.95, 22.12, 191.80, 154.60],
            [19.66, 36.05, 118.97, 16.09, 224.52, 154.64],
            [4.66, 37.43, 117.75, -36.21, 214.95, 155.44],
            [15.44, 40.27, 111.21, -0.01, 211.65, 157.86],
            [22.05, 28.27, 136.62, 13.98, 218.45, 157.86],
            [4.63, 28.95, 132.45, -39.26, 220.45, 157.28],
            [9.37, 40.17, 107.15, -23.39, 213.09, 156.80],
            [18.78, 44.36, 101.57, 5.30, 205.47, 145.54],
            [26.34, 36.25, 115.69, 27.54, 207.70, 146.91],
            [15.14, 29.50, 121.26, -10.89, 234.88, 146.89],
            [4.53, 41.82, 105.06, -42.38, 207.62, 148.63],
            [20.02, 42.79, 105.01, 8.38, 197.41, 148.80],
            [28.53, 26.65, 133.86, 18.93, 224.70, 148.80],
            [14.48, 16.79, 136.85, -23.08, 247.10, 143.89],
            [14.12, 29.56, 119.95, -16.60, 247.18, 143.89],
            [18.35, 38.52, 107.51, 14.21, 243.56, 139.11],
            [10.72, 29.37, 116.58, -35.60, 244.81, 143.94],
            [21.69, 37.33, 109.55, 21.70, 221.00, 144.44],
            [23.32, 24.06, 131.89, 9.99, 232.03, 160.57],
            [13.40, 43.54, 100.36, -7.56, 203.79, 160.56],
            [21.64, 30.78, 123.46, 22.26, 226.13, 160.55],
            [7.70, 31.26, 120.52, -33.88, 226.18, 160.58],
            [6.24, 37.03, 117.82, -35.20, 202.70, 160.46],
            [19.64, 33.05, 132.47, -3.43, 195.42, 154.59]
        ]
        
        waypoints_cam2 = [
            [35.14, 46.86, 87.19, 87.99, 224.15, 121.78],
            [35.26, 43.95, 85.73, 92.30, 231.30, 137.05],
            [34.96, 48.33, 86.27, 64.58, 227.12, 127.91],
            [42.66, 39.45, 94.87, 86.39, 228.47, 109.77],
            [42.63, 33.74, 96.34, 109.63, 234.86, 114.52],
            [39.19, 37.75, 90.30, 103.13, 234.66, 149.02],
            [42.68, 31.75, 95.02, 117.37, 238.63, 149.59],
            [37.33, 38.74, 102.01, 78.57, 227.17, 149.55],
            [33.37, 40.19, 93.90, 99.17, 228.18, 107.27],
            [30.16, 43.74, 93.45, 77.95, 228.35, 110.60],
            [24.56, 52.42, 72.47, 72.71, 223.64, 103.41],
            [24.55, 51.35, 65.58, 91.18, 224.77, 103.95],
            [44.33, 27.55, 106.94, 100.36, 245.30, 106.76],
            [44.12, 29.55, 112.19, 81.89, 242.79, 106.85],
            [33.91, 43.76, 84.67, 74.86, 228.07, 106.85],
            [31.33, 53.15, 55.75, 91.19, 243.16, 106.85],
            [29.21, 52.34, 56.08, 106.74, 241.29, 94.15],
            [38.81, 37.44, 82.67, 115.34, 240.63, 148.92],
            [38.84, 32.38, 94.48, 113.60, 236.77, 139.39],
            [37.53, 32.77, 105.33, 102.87, 229.39, 113.11],
            [57.33, 31.33, 105.65, 101.54, 225.22, 126.29],
            [30.17, 46.85, 83.44, 70.34, 229.62, 126.07],
            [30.30, 47.59, 67.88, 99.82, 235.88, 110.41],
            [43.47, 47.83, 66.88, 100.20, 235.64, 110.41],
            [42.87, 56.16, 68.12, 72.95, 224.25, 100.10],
            [35.14, 46.86, 87.19, 87.99, 224.15, 121.78]
        ]

        velj = 20.0 
        accj = 20.0

        unix_timestamp = int(time.time())
        base_output_folder = f"calibration_data_{unix_timestamp}"
        os.makedirs(base_output_folder, exist_ok=True)
        print(f"Speichere Daten in: {base_output_folder}/")

        ctx = rs.context()
        devices = ctx.query_devices()
        num_cameras = len(devices)
        
        if num_cameras == 0:
            print("Fehler: Keine RealSense Kameras gefunden!")
            return
            
        for idx, device in enumerate(devices):
            cam_idx = idx + 1
            serial_number = device.get_info(rs.camera_info.serial_number)
            
            cam_folder = os.path.join(base_output_folder, f"camera_{cam_idx}")
            os.makedirs(cam_folder, exist_ok=True)

            with open(os.path.join(cam_folder, "serial.txt"), "w") as f:
                f.write(serial_number)
            
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            
            try:
                pipeline.start(config)
            except Exception as e:
                print(f"Konnte Kamera {cam_idx} nicht starten: {e}")
                continue

            print(f"\n--- Starte Kalibrierungslauf für Kamera {cam_idx}/{num_cameras} ---")
            
            if cam_idx == 1:
                current_waypoints = waypoints_cam1
            elif cam_idx == 2:
                current_waypoints = waypoints_cam2
            else:
                print(f"Keine Wegpunkte für Kamera {cam_idx} definiert. Überspringe...")
                pipeline.stop()
                continue
            
            for i, target_joints in enumerate(current_waypoints):
                print(f"Fahre Pose {i+1}/{len(current_waypoints)} an...")
                
                movej(target_joints, velj, accj)
                time.sleep(0.5) 

                aktuelle_pose = extract_pose(get_current_posx(DR_BASE)[0])

                for _ in range(5):
                    pipeline.wait_for_frames() 
                    
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    bild_pfad = os.path.join(cam_folder, f"cam_{i:03d}.png")
                    pose_pfad = os.path.join(cam_folder, f"pose_{i:03d}.npy")
                    
                    cv2.imwrite(bild_pfad, frame)
                    np.save(pose_pfad, np.array(aktuelle_pose))
                    print(f"-> Pose {i+1} erfolgreich gespeichert.")
                else:
                    print(f"-> ERROR bei Pose {i+1}: Kein Bild empfangen.")

            pipeline.stop()
            
            print(f"Fahre zurück auf Startposition für Kamera {cam_idx}...")

        print("\nAlle Kameras wurden abgearbeitet.")

    except Exception as e:
        print(f"\n[KRITISCHER FEHLER IM BEWEGUNGS-THREAD]: {e}")
        traceback.print_exc()
    finally:
        movej([19.64, 33.05, 132.47, -3.43, 195.42, 154.59], velj, accj)
        movement_finished_event.set()

if __name__ == "__main__":
    main()