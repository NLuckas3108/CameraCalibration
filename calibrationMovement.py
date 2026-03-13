import rclpy
import threading
import rclpy.executors
import os
import time
import random
import cv2        
import numpy as np
import pyrealsense2 as rs
import traceback 

from dsr_msgs2.srv import SetRobotMode, MoveStop

ROBOT_ID   = "dsr01"
ROBOT_MODEL= "m1013"

import DR_init
DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def set_robot_mode_srv(node, mode):
    """Schaltet zwischen Manual (0) und Autonomous (1) um."""
    client = node.create_client(SetRobotMode, f'/{ROBOT_ID}/system/set_robot_mode')
    if not client.wait_for_service(timeout_sec=3.0):
        print(f"Fehler: Service '/{ROBOT_ID}/system/set_robot_mode' nicht erreichbar!")
        return False
        
    req = SetRobotMode.Request()
    req.robot_mode = mode 
    future = client.call_async(req)
    
    timeout = 2.0 
    start_time = time.time()
    while not future.done() and (time.time() - start_time) < timeout:
        time.sleep(0.05)
        
    return True


def call_move_stop(node):
    """Feuert einen sofortigen Stopp-Befehl ab, um Alarme zu verhindern."""
    client = node.create_client(MoveStop, f'/{ROBOT_ID}/motion/move_stop')
    if client.wait_for_service(timeout_sec=1.0):
        req = MoveStop.Request()
        req.stop_mode = 1
        client.call_async(req)
        return True
    return False

def extract_pose(raw_pose_data):
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
        from DSR_ROBOT2 import movel, get_current_posx, DR_BASE
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
    from DSR_ROBOT2 import get_current_posj

    try:
        velx = [50, 30]
        accx = [50, 30]
            
        delta_x = [-50, 0, 50]
        delta_y = [-50, 0, 50]
        delta_z = [-50, 0, 50]
        delta_alpha = [-10, -5, 0, 5, 10]
        delta_beta = [-10, -5, 0, 5, 10]
        delta_gamma = [-10, -5, 0, 5, 10]

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
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            try:
                pipeline.start(config)
            except Exception as e:
                print(f"Konnte Kamera {cam_idx} nicht starten: {e}")
                continue

            print(f"\n--- Kamera {cam_idx}/{num_cameras} ---")
            print("Schalte Roboter in den MANUELLEN Modus für Direct Teaching...")
            set_robot_mode_srv(node, 0)

            print("Bitte den Roboter nun per Knopf auf die Kamera ausrichten und ENTER drücken.")
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue
                    
                frame_img = np.asanyarray(color_frame.get_data())
                cv2.putText(frame_img, f"Kamera {cam_idx}: Ausrichten und ENTER druecken", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Livefeed Kalibrierung", frame_img)
                
                if cv2.waitKey(1) in [13, 32]:
                    break
                    
            cv2.destroyAllWindows()
            cv2.waitKey(1) 
            
            print("Schalte Roboter zurück in den AUTONOMEN Modus...")
            set_robot_mode_srv(node, 1)
            time.sleep(1.0) 
            
            base_pose = extract_pose(get_current_posx(DR_BASE)[0])
            print(f"Grundpose gespeichert: {base_pose}")
            print("Starte 50 Posen...")
            
            for i in range(50):
                target_reached = False
                while not target_reached:
                    target_pos = [
                        base_pose[0] + random.choice(delta_x),
                        base_pose[1] + random.choice(delta_y),
                        base_pose[2] + random.choice(delta_z),
                        base_pose[3] + random.choice(delta_alpha),
                        base_pose[4] + random.choice(delta_beta),
                        base_pose[5] + random.choice(delta_gamma)
                    ]

                    def move_worker():
                        try:
                            movel(target_pos, velx, accx)
                        except Exception:
                            pass

                    move_thread = threading.Thread(target=move_worker)
                    move_thread.start()
                    
                    limit_hit = False
                    
                    while move_thread.is_alive():
                        try:
                            cur_j = get_current_posj()
                            if abs(cur_j[1]) > 90.0: 
                                call_move_stop(node)
                                print(f"-> SOFT-LIMIT RETTUNG: Joint 2 bei {cur_j[1]:.1f} Grad. Notbremsung!")
                                limit_hit = True
                                break
                        except Exception:
                            pass
                        
                        time.sleep(0.05) 
                    
                    move_thread.join(timeout=1.0)
                    
                    if limit_hit:
                        time.sleep(1.0)
                        continue

                    aktuelle_pose = extract_pose(get_current_posx(DR_BASE)[0])
                    
                    distanz = np.linalg.norm(np.array(aktuelle_pose[:3]) - np.array(target_pos[:3]))
                    
                    if distanz < 5.0:
                        target_reached = True
                    else:
                        print(f"Pose unerreichbar (Abweichung: {distanz:.1f} mm). Würfele neu...")

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
                    print(f"Kamera {cam_idx} - Pos {i+1}/50: Bild und Pose erfolgreich gespeichert.")
                else:
                    print(f"ERROR bei Kamera {cam_idx} - Pos {i+1}: Kein Bild empfangen.")

            pipeline.stop()
            movel(base_pose, velx, accx)

        print("\nAlle Kameras wurden abgearbeitet.")

    except Exception as e:
        print(f"\n[KRITISCHER FEHLER IM BEWEGUNGS-THREAD]: {e}")
        traceback.print_exc()
    finally:
        movement_finished_event.set()

if __name__ == "__main__":
    main()