import rclpy
import threading
import rclpy.executors
import time
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

# Service für den Moduswechsel importieren
from dsr_msgs2.srv import SetRobotMode

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


def extract_joints(raw_joint_data):
    """Extrahiert die 6 Gelenkwinkel sauber aus der Doosan-Liste."""
    result = []
    for item in raw_joint_data:
        if isinstance(item, (list, tuple)):
            result.append(float(item[0]))
        else:
            result.append(float(item))
    return result[:6]


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('dsr_logger_node', namespace=ROBOT_ID)
    DR_init.__dsr__node = node
    
    try:
        from DSR_ROBOT2 import get_current_posj
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        return
        
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    print("\n--- Wegpunkt-Logger mit Live-View gestartet ---")
    
    # RealSense Kameras initialisieren
    ctx = rs.context()
    devices = ctx.query_devices()
    num_cameras = len(devices)
    
    if num_cameras == 0:
        print("Fehler: Keine RealSense Kameras gefunden. Beende.")
        sys.exit(1)
        
    pipelines = []
    for idx, device in enumerate(devices):
        serial_number = device.get_info(rs.camera_info.serial_number)
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipe.start(config)
        pipelines.append((idx + 1, pipe))
        print(f"Kamera {idx + 1} (SN: {serial_number}) gestartet.")

    # Roboter für Direct Teaching freigeben
    print("\nSchalte Roboter in den MANUELLEN Modus für Direct Teaching...")
    set_robot_mode_srv(node, 0)
    time.sleep(1.0)
    
    print("\n" + "="*50)
    print("BEDIENUNG:")
    print("1. Bewege den Roboter per Hand an die gewünschte Position.")
    print("2. KLICKE IN EINES DER KAMERAFENSTER, damit es aktiv ist.")
    print("3. Drücke ENTER, um die aktuellen Gelenkwinkel zu loggen.")
    print("4. Drücke 'q', um das Skript zu beenden.")
    print("="*50 + "\n")
    
    try:
        while True:
            # Kamerabilder abrufen und anzeigen
            for cam_idx, pipe in pipelines:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    img = np.asanyarray(color_frame.get_data())
                    cv2.putText(img, f"Kamera {cam_idx}: ENTER = Loggen | q = Beenden", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(f"Livefeed Kamera {cam_idx}", img)
            
            # Tastaturabfrage über OpenCV (wartet 1 ms)
            key = cv2.waitKey(1) & 0xFF
            
            # ENTER-Taste (ASCII 13)
            if key == 13: 
                raw_joints = get_current_posj()
                joints = extract_joints(raw_joints)
                
                formatted_str = f"            [{joints[0]:.2f}, {joints[1]:.2f}, {joints[2]:.2f}, {joints[3]:.2f}, {joints[4]:.2f}, {joints[5]:.2f}],"
                
                print("\nFüge diese Zeile in deine Liste ein:")
                print(formatted_str)
                print("-" * 50)
                
            # 'q'-Taste zum Beenden
            elif key == ord('q'):
                print("\nBeenden angefordert...")
                break
                
    except KeyboardInterrupt:
        pass
        
    # Kameras und Fenster schließen
    print("Schließe Kamera-Streams...")
    for _, pipe in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()
    
    # Roboter wieder sperren
    print("Schalte Roboter zurück in den AUTONOMEN Modus...")
    set_robot_mode_srv(node, 1)
    time.sleep(1.0)
        
    print("Beende Logger...")
    executor.shutdown()
    executor_thread.join()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()