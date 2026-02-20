import pyrealsense2 as rs
import numpy as np

# Pipeline konfigurieren und starten
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 

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

    print("Kameramatrix:")
    print(repr(camera_matrix))
    print("\nVerzerrungskoeffizienten:")
    print(repr(dist_coeffs))

finally:
    pipeline.stop()