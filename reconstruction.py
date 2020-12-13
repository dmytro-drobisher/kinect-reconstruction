import cv2
import numpy as np
import open3d
import argparse
import os
import kinect_recorder
from opencv_pose_estimation import pose_estimation

# Command line arguments
parser = argparse.ArgumentParser(description='Kinect video recorder')
parser.add_argument("-record", default=False, help="location of depth frames")
parser.add_argument("-colour-path", default="/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/colour", help="location of colour frames")
parser.add_argument("-depth-path", default="/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/depth", help="location of depth frames")
parser.add_argument("-fragment-path", default="/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/fragments", help="location of computed fragments")
parser.add_argument("-pose-path", default="/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/poses", help="location of depth frames")

# Estimate transformation between two RGBD images
def compute_odometry(sid, tid, source, target, intrinsic):
    odo_option = open3d.odometry.OdometryOption()
    odo_option.min_depth = 0.5
    odo_option.max_depth = 4.0
    odo_option.iteration_number_per_pyramid_level = open3d.utility.IntVector([15, 7, 4])
    
    if tid == sid + 1:
        print(sid, tid)

        [success_hybrid_term, odometry_estimate, info] = open3d.odometry.compute_rgbd_odometry(source, target, intrinsic, np.identity(4), open3d.odometry.RGBDOdometryJacobianFromHybridTerm(), odo_option)
        return success_hybrid_term, odometry_estimate, info
    else:
        success, odo_init = pose_estimation(frames_rgbd_colour[s - fragment[0]], frames_rgbd_colour[t - fragment[0]], kinect_intrinsic, False)

        if not success:
            return False, np.identity(4), np.identity(6)

        print(sid, tid)

        [success_hybrid_term, odometry_estimate, info] = open3d.odometry.compute_rgbd_odometry(source, target, intrinsic, odo_init, open3d.odometry.RGBDOdometryJacobianFromHybridTerm(), odo_option)
        return success_hybrid_term, odometry_estimate, info

def read_data(colour_path, depth_path):
    colour = list()
    depth = list()
    colour_names = os.listdir(colour_path)
    depth_names = os.listdir(depth_path)

    colour_names.sort()
    depth_names.sort()

    if len(colour_names) != len(depth_names):
        return colour, depth
    
    for filename in zip(colour_names, depth_names):
        print(filename[0])
        rgb = cv2.imread("".join([colour_path, "/", filename[0]]))
        colour.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth.append(np.load("".join([depth_path, "/", filename[1]])))
    
    return colour, depth

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Info)

# Camera Intrinsics
image_width = 640
image_height = 480
focal_length = 531.15   #pixels
kinect_intrinsic = open3d.camera.PinholeCameraIntrinsic(image_width, image_height, focal_length, focal_length, image_width // 2, image_height // 2)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.record:
        colour, depth = kinect_recorder.record()
    else:
        colour, depth = read_data(args.colour_path, args.depth_path)

    cv2.destroyAllWindows()

    # fragment start and end ids
    fragment_start = [100 * i for i in range(1, (len(colour) // 100) + 1)]
    fragment_end = [100 * i for i in range(2, len(colour) // 100 + 1)] + [len(colour)]

    # list of computed fragments
    fragments = list()

    # create directory to store fragments
    if not os.path.exists(args.fragment_path):
        os.mkdir(args.fragment_path)

    # create posegraph directory
    if not os.path.exists(args.pose_path):
        os.mkdir(args.pose_path)

    currentFragment = 1
    for fragment in zip(fragment_start, fragment_end):
        # Reconstruction loop
        odometry = np.identity(4)
        point_cloud = open3d.geometry.PointCloud()
        pose_graph = open3d.registration.PoseGraph()
        pose_graph.nodes.append(open3d.registration.PoseGraphNode(odometry))

        # precompute rgbd images to save time
        frames_rgbd = [open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(cv2.cvtColor(colour[i],      cv2.COLOR_BGR2RGB)), open3d.geometry.Image(depth[i]), depth_scale=1000., convert_rgb_to_intensity=True) for i in range(fragment[0], fragment[1])  ]

        frames_rgbd_colour = [open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(colour[i]), open3d.geometry.Image(depth[i]), depth_scale=1000.0, convert_rgb_to_intensity=False) for i in range(fragment[0], fragment[1])]

        # build pose graph
        for s in range(fragment[0], fragment[1]):
            for t in range(s + 1, fragment[1]):
                if t == s + 1:
                    print("Fragment %04d: " % len(fragments), s - fragment[0], " : ", t - fragment[0])
                    success_hybrid_term, odometry_estimate, info = compute_odometry(s, t, frames_rgbd[s - fragment[0]], frames_rgbd[t - fragment[0]], kinect_intrinsic)

                    odometry = np.dot(odometry_estimate, odometry)
                    pose_graph.nodes.append(open3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(open3d.registration.PoseGraphEdge(s - fragment[0], t - fragment[0], odometry_estimate, info, uncertain=False))

                if s % 5 == 0 and t % 5 == 0:
                    print("Fragment %04d: " % len(fragments), s - fragment[0], " : ", t - fragment[0])

                    success_hybrid_term, odometry_estimate, info = compute_odometry(s, t, frames_rgbd[s - fragment[0]], frames_rgbd[t - fragment[0]], kinect_intrinsic)
                    if success_hybrid_term:
                        pose_graph.edges.append(open3d.registration.PoseGraphEdge(s - fragment[0], t - fragment[0], odometry_estimate, info, uncertain=True))

        open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)

        # pose graph optimisation
        method = open3d.registration.GlobalOptimizationLevenbergMarquardt()
        criteria = open3d.registration.GlobalOptimizationConvergenceCriteria()
        option = open3d.registration.GlobalOptimizationOption(max_correspondence_distance=0.01, edge_prune_threshold=0.25,
            preference_loop_closure=0.1,
            reference_node=0)

        open3d.registration.global_optimization(pose_graph, method, criteria, option)
        volume = open3d.integration.ScalableTSDFVolume(voxel_length=1.2 / 512.0, sdf_trunc=0.02, color_type=open3d.integration.TSDFVolumeColorType.RGB8)

        # construct point cloud
        for i in range(fragment[0], fragment[1]):
            #pcd = open3d.geometry.PointCloud.create_from_rgbd_image(frames_rgbd_colour[i - fragment[0]], kinect_intrinsic)
            #pcd = open3d.geometry.PointCloud.voxel_down_sample(pcd, 0.002)
            #pcd.transform(pose_graph.nodes[i - fragment[0]].pose)
            #point_cloud += pcd
            #point_cloud = open3d.geometry.PointCloud.voxel_down_sample(point_cloud, 0.002)
            volume.integrate(frames_rgbd_colour[i - fragment[0]], kinect_intrinsic, np.linalg.inv(pose_graph.nodes[i - fragment[0]].pose))
        
        point_cloud = volume.extract_point_cloud()
        point_cloud.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

        open3d.io.write_point_cloud("".join([args.fragment_path, "/%08d.ply" % currentFragment]), point_cloud)
        open3d.io.write_pose_graph("".join([args.pose_path, "/%08d.json" % currentFragment]), pose_graph)
        open3d.visualization.draw_geometries([point_cloud])
        #fragments.append(point_cloud)
        currentFragment += 1

        #mesh = volume.extract_triangle_mesh()
        #open3d.visualization.draw_geometries([mesh])
        #open3d.visualization.draw_geometries([point_cloud])
