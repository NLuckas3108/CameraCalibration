import pyrealsense2 as rs
import numpy as np

def get_intrinsics(serial_number, width=640, height=480):
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    
    pipeline_profile = pipeline.start(config)
    stream_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream_profile.get_intrinsics()
    
    pipeline.stop()
    
    camera_matrix = np.array([
        [intrinsics.fx, 0,             intrinsics.ppx],
        [0,             intrinsics.fy, intrinsics.ppy],
        [0,             0,             1]
    ], dtype=np.float64)
    
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
    
    return camera_matrix, dist_coeffs