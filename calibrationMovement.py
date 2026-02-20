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
import cv2        
import numpy as np  

ROBOT_ID   = "dsr01"
ROBOT_MODEL= "m1013"

global movement_finished

import DR_init
DR_init.__dsr__id   = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    global movement_finished
    movement_finished = False
    
    rclpy.init(args=args)

    node = rclpy.create_node('dsr_example_demo_py', namespace=ROBOT_ID)

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

    movement_thread = threading.Thread(
        target=run_movement,
        args=(node, movel, get_current_posx, DR_BASE)
    )
    movement_thread.start()

    while rclpy.ok() and not movement_finished:
        time.sleep(0.1)

    node.get_logger().info("Fahre Executor herunter...")
    executor.shutdown()
    executor_thread.join()
    node.destroy_node()
    rclpy.shutdown()
    
def run_movement(node, movel, get_current_posx, DR_BASE):
    velx = [50, 30]
    accx = [50, 30]
        
    x_values = [750, 850, 950]
    y_values = [150, 250, 350]
    z_values = [300, 400, 500]
    rot_values_alpha = [-20, -10, 0, 10, 20]
    rot_values_beta = [70, 80, 90, 100, 110]
    rot_values_gamma = [-140, -130, -120, -110, -100]

    orientation = [0, 90, -120]

    output_folder = "calibration_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(4)
    for _ in range(50):
        cap.read()

    movel([750, 150, 300, 0, 90, -120], velx, accx)
    
    for i in range (0, 50):
        zufalls_pos = [random.choice(x_values), random.choice(y_values), random.choice(z_values), random.choice(rot_values_alpha), random.choice(rot_values_beta), random.choice(rot_values_gamma)]
        
        movel(zufalls_pos, velx, accx)
        
        time.sleep(2)
        
        aktuelle_pose = get_current_posx(DR_BASE)[0]
        
        for _ in range(5):
            cap.read()
        
        ret, frame = cap.read()
        
        if ret:
            bild_pfad = os.path.join(output_folder, f"cam_{i:03d}.png")
            pose_pfad = os.path.join(output_folder, f"pose_{i:03d}.npy")
            
            cv2.imwrite(bild_pfad, frame)
            
            np.save(pose_pfad, np.array(aktuelle_pose))
            
            print(f"Position {i+1}/50: image and pose were saved successfully")
        else:
            print(f"ERROR at position {i+1}: camera couldnt get image")

    movel([750, 150, 300, 0, 90, -120], velx, accx)
    
    cap.release()
    
    global movement_finished
    movement_finished = True

if __name__ == "__main__":
        main()