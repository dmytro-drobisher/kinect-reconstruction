import freenect
import cv2
import numpy as np
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

# Camera Intrinsics
image_width = 640
image_height = 480
focal_length = 531.15   #pixels
kinect_intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, focal_length, focal_length, image_width // 2, image_height // 2)

# Visualiser
visualiser = o3d.visualization.Visualizer()
visualiser.create_window()
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.array([[i, i, i] for i in range(-5, 5)]))
visualiser.add_geometry(point_cloud)

# Windows for display
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

# A list of captured rgbd frames
frames_rgbd = []
frames_pcd = []

# Capture loop
capture = True

while capture:
    rgb = freenect.sync_get_video(index=0)[0]
    depth = freenect.sync_get_depth(index=0, format=freenect.DEPTH_REGISTERED)[0]
    
    
    rgbd = o3d.create_rgbd_image_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth.astype(np.float32)), depth_scale=1000, convert_rgb_to_intensity=True)

    rgbd_colour = o3d.create_rgbd_image_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth.astype(np.float32)), depth_scale=1000, convert_rgb_to_intensity=False)
    
    pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_colour, kinect_intrinsic)
    
    point_cloud.points = pcd.points
    point_cloud.colors = pcd.colors
    #o3d.draw_geometries([pcd])
    #visualiser.update_geometry()
    #visualiser.poll_events()
    #visualiser.update_renderer()

    cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    cv2.imshow("Depth", depth)
    key = cv2.waitKey(1)

    if key == ord(" "):
        print("Frame captured")
        frames_rgbd.append(rgbd)
        frames_pcd.append(pcd)
    elif key == ord("x"):
        capture = False

# Reconstruction loop
odo_option = o3d.odometry.OdometryOption()
odo_option.min_depth = 0.5
odo_option.max_depth = 4.0
odo_option.iteration_number_per_pyramid_level = o3d.utility.IntVector([15, 7, 4])

point_cloud = frames_pcd[0]
prev_odometry = np.identity(4)

for i in range(len(frames_rgbd) - 2):
    [success_hybrid_term, odometry_estimate, info] = o3d.odometry.compute_rgbd_odometry(frames_rgbd[i], frames_rgbd[i + 1], kinect_intrinsic, prev_odometry, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), odo_option)

    if success_hybrid_term:
        point_cloud.transform(odometry_estimate)
        point_cloud += frames_pcd[i + 1]


        prev_odometry = odometry_estimate

point_cloud = o3d.geometry.voxel_down_sample(point_cloud, 0.008)

o3d.visualization.draw_geometries([point_cloud])
o3d.io.write_point_cloud("pcd.ply", point_cloud)