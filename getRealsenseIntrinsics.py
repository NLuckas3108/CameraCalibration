import pyrealsense2 as rs
import numpy as np

def get_intrinsics(width=640, height=480):
    pipeline = rs.pipeline()
    config = rs.config()
    

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30) 

    profile = pipeline.start(config)

    try:
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        camera_matrix = np.array([
            [intrinsics.fx, 0,             intrinsics.ppx],
            [0,             intrinsics.fy, intrinsics.ppy],
            [0,             0,             1]
        ])
        
        dist_coeffs = np.array(intrinsics.coeffs)

        return camera_matrix, dist_coeffs

    finally:
        pipeline.stop()