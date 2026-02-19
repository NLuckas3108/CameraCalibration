import rclpy
import threading
import rclpy.executors
import os
import sys 
import time
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import csv
from datetime import datetime
import random
import cv2          # Neu für die Kamera
import numpy as np  # Neu für das Speichern der Pose

# for single robot
ROBOT_ID   = "dsr01"
ROBOT_MODEL= "m1013"

global movement_finished

# Import und Initialisierung des DSR-spezifischen Initialisierungsmoduls
import DR_init
DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    global movement_finished
    movement_finished = False
    
    rclpy.init(args=args)

    # Hauptnode
    node = rclpy.create_node('dsr_example_demo_py', namespace=ROBOT_ID)

    # Import Roboter-Funktionen
    DR_init.__dsr__node = node
    try:
        from DSR_ROBOT2 import print_ext_result, movej, movel, movec, move_periodic, move_spiral, set_velx, set_accx, get_current_posj, get_current_posx, get_current_pose, get_desired_posx, DR_BASE, DR_TOOL, DR_AXIS_X, DR_MV_MOD_ABS
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        return
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # get_current_posx und DR_BASE mit in den Thread übergeben
    movement_thread = threading.Thread(
        target=run_movement,
        args=(node, movel, get_current_posx, DR_BASE)
    )
    movement_thread.start()
    
    # Warten auf Beendigung der Bewegung
    while rclpy.ok() and not movement_finished:
        time.sleep(0.1)

    # Shutdown-Logik
    node.get_logger().info("Fahre Executor herunter...")
    executor.shutdown()
    executor_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    
def run_movement(node, movel, get_current_posx, DR_BASE):
    velx = [150, 90]
    accx = [150, 90]
        
    x_values = [700, 800, 900]
    y_values = [150, 250, 350]
    z_values = [300, 400, 500]
    rot_values_alpha = [-20, -10, 0, 10, 20]
    rot_values_beta = [70, 80, 90, 100, 110]
    rot_values_gamma = [-140, -130, -120, -110, -100]

    orientation = [0, 90, -120]

    point_array = []

    for z in z_values:
        for y in y_values:
            for x in x_values:
                punkt = [x, y, z] + orientation
                point_array.append(punkt)
            
    # --- NEU: Ordner für Kalibrierungsdaten erstellen ---
    ausgabe_ordner = "calibration_data"
    if not os.path.exists(ausgabe_ordner):
        os.makedirs(ausgabe_ordner)

    cap = cv2.VideoCapture(4)
    time.sleep(10)

    movel(point_array[0], velx, accx)
    
    # Auf 50 Iterationen erhöht
    for i in range (0, 50):
        x = random.choice(x_values)
        y = random.choice(y_values)
        z = random.choice(z_values)
        zufalls_pos = [x, y, z, random.choice(rot_values_alpha), random.choice(rot_values_beta), random.choice(rot_values_gamma)]
        
        # 1. Roboter fährt zur Zufallspose
        movel(zufalls_pos, velx, accx)
        
        # 2. WICHTIG: Halbe Sekunde warten, damit der Greifer nicht mehr wackelt
        time.sleep(2)
        
        # 3. Exakte Pose abfragen (im Base-Koordinatensystem)
        aktuelle_pose = get_current_posx(DR_BASE)[0]
        
        # 4. Bild aufnehmen
        ret, frame = cap.read()
        
        if ret:
            bild_pfad = os.path.join(ausgabe_ordner, f"cam_{i:03d}.png")
            pose_pfad = os.path.join(ausgabe_ordner, f"pose_{i:03d}.npy")
            
            cv2.imwrite(bild_pfad, frame)
            
            np.save(pose_pfad, np.array(aktuelle_pose))
            
            print(f"Position {i+1}/50: Bild und Pose erfolgreich gespeichert.")
        else:
            print(f"FEHLER an Position {i+1}: Kamera konnte kein Bild abgreifen!")

    movel(point_array[0], velx, accx)
    
    cap.release()
    
    global movement_finished
    movement_finished = True

if __name__ == "__main__":
        main()