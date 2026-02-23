import rclpy
import threading
import rclpy.executors
import os
import time
from rclpy.node import Node
import csv
import random
import cv2        
import numpy as np
import pyrealsense2 as rs

ROBOT_ID   = "dsr01"
ROBOT_MODEL= "m1013"

import DR_init
DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    movement_finished_event = threading.Event()
    
    rclpy.init(args=args)

    node = rclpy.create_node('dsr_example_demo_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node
    try:
        from DSR_ROBOT2 import movej, movel, get_current_posx, DR_BASE
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        return
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    movement_thread = threading.Thread(
        target=run_movement,
        args=(node, movel, get_current_posx, DR_BASE, movement_finished_event)
    )
    movement_thread.start()

    while rclpy.ok() and not movement_finished_event.is_set():
        time.sleep(0.1)

    node.get_logger().info("Fahre Executor herunter...")
    executor.shutdown()
    executor_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    
def run_movement(node, movel, get_current_posx, DR_BASE, movement_finished_event):
    velx = [50, 30]
    accx = [50, 30]
        
    delta_x = [-100, 0, 100]
    delta_y = [-100, 0, 100]
    delta_z = [-100, 0, 100]
    delta_alpha = [-20, -10, 0, 10, 20]
    delta_beta = [-20, -10, 0, 10, 20]
    delta_gamma = [-20, -10, 0, 10, 20]

    unix_timestamp = int(time.time())
    base_output_folder = f"calibration_data_{unix_timestamp}"
    os.makedirs(base_output_folder, exist_ok=True)
    print(f"Speichere Daten in: {base_output_folder}/")

    ctx = rs.context()
    devices = ctx.query_devices()
    num_cameras = len(devices)
    
    if num_cameras == 0:
        print("Fehler: Keine RealSense Kameras gefunden!")
        movement_finished_event.set()
        return
        
    print(f"Es wurden {num_cameras} RealSense Kamera(s) gefunden.")

    for idx, device in enumerate(devices):
        cam_idx = idx + 1
        serial_number = device.get_info(rs.camera_info.serial_number)
        
        cam_folder = os.path.join(base_output_folder, f"camera_{cam_idx}")
        os.makedirs(cam_folder, exist_ok=True)
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            pipeline.start(config)
        except Exception as e:
            print(f"Konnte Kamera {cam_idx} (SN: {serial_number}) nicht starten: {e}")
            continue

        print(f"\n--- Kamera {cam_idx}/{num_cameras} ---")
        print("Bitte den Roboter auf die Kamera ausrichten.")
        print("Fokus auf das Live-Fenster setzen und 'ENTER' drücken, wenn die Position passt.")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            frame_img = np.asanyarray(color_frame.get_data())
            cv2.putText(frame_img, f"Kamera {cam_idx}: Ausrichten und ENTER druecken", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Livefeed Kalibrierung", frame_img)
            
            key = cv2.waitKey(1)
            if key in [13, 32]:
                break
                
        cv2.destroyAllWindows()
        
        base_pose = get_current_posx(DR_BASE)[0]
        print(f"Grundpose gespeichert: {base_pose}")
        
        for i in range(50):
            target_pos = [
                base_pose[0] + random.choice(delta_x),
                base_pose[1] + random.choice(delta_y),
                base_pose[2] + random.choice(delta_z),
                base_pose[3] + random.choice(delta_alpha),
                base_pose[4] + random.choice(delta_beta),
                base_pose[5] + random.choice(delta_gamma)
            ]
            
            movel(target_pos, velx, accx)
            time.sleep(2)
            
            aktuelle_pose = get_current_posx(DR_BASE)[0]
            
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
                print(f"Kamera {cam_idx} - Pos {i+1}/50: Bild und Pose gespeichert.")
            else:
                print(f"ERROR bei Kamera {cam_idx} - Pos {i+1}: Kein Bild empfangen.")

        pipeline.stop()
        
        movel(base_pose, velx, accx)

    print("\nAlle Kameras wurden abgearbeitet.")
    movement_finished_event.set()

if __name__ == "__main__":
    main()