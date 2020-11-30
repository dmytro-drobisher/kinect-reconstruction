import os
import open3d
import numpy as np

ROOT = "/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/fragments/"
POSEGRAPH = "/media/dmytro/Storage/Ubuntu/Documents/Projects/Kinect/poses/"

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)

def register_fragments(source, target, trans_init=np.identity(4)):
    print("Registering: %d and %d" % (source, target))

    pcd_source = open3d.io.read_point_cloud(ROOT + files[source])
    pcd_target = open3d.io.read_point_cloud(ROOT + files[target])

    pcd_source_down = open3d.geometry.PointCloud.voxel_down_sample(pcd_source, 0.02)
    pcd_target_down = open3d.geometry.PointCloud.voxel_down_sample(pcd_target, 0.02)

    fpfh_source = open3d.registration.compute_fpfh_feature(pcd_source_down, open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    fpfh_target = open3d.registration.compute_fpfh_feature(pcd_target_down, open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    if source + 1 != target:
        res = open3d.registration.registration_ransac_based_on_feature_matching(pcd_source_down, pcd_target_down, fpfh_source, fpfh_target, 0.01, open3d.registration.TransformationEstimationPointToPoint(False), 4, [open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), open3d.registration.CorrespondenceCheckerBasedOnDistance(0.01)], open3d.registration.RANSACConvergenceCriteria(4000000, 500))
        trans_init = res.transformation
        print(res)

    pcd_source_down = open3d.geometry.PointCloud.voxel_down_sample(pcd_source, 0.005)
    pcd_target_down = open3d.geometry.PointCloud.voxel_down_sample(pcd_target, 0.005)

    res = open3d.registration.registration_icp(pcd_source_down, pcd_target_down, 0.01, trans_init, open3d.registration.TransformationEstimationPointToPlane(), open3d.registration.ICPConvergenceCriteria(max_iteration=200))
    information = open3d.registration.get_information_matrix_from_point_clouds(pcd_source, pcd_target, 0.01, res.transformation)

    #pcd_source.transform(res.transformation)
    #open3d.visualization.draw_geometries([pcd_source, pcd_target])
    print(res.transformation)
    
    return res.transformation, information
    #return np.dot(trans_init, res.transformation), information

if __name__ == "__main__":
    files = os.listdir(ROOT)
    files.sort()

    if len(files) <= 1:
        exit(0)
    
    pose_graph = open3d.registration.PoseGraph()
    method = open3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = open3d.registration.GlobalOptimizationConvergenceCriteria()
    option = open3d.registration.GlobalOptimizationOption(max_correspondence_distance=0.01, edge_prune_threshold=0.25,
        preference_loop_closure=0.2,
        reference_node=0)

    pose = np.identity(4)
    pose_graph.nodes.append(open3d.registration.PoseGraphNode(np.linalg.inv(pose)))

    # Assume pose graphs are in order and initialise the posegraph with sequantial registrations
    #for i in range(3):
    for i in range(len(files) - 1):
        pose_graph_source = open3d.io.read_pose_graph(POSEGRAPH + files[i][:-4] + ".json")

        transformation, information = register_fragments(i, i + 1, np.linalg.inv(list(pose_graph_source.nodes)[-1].pose))
    
        pose = np.dot(transformation, pose)
        pose_graph.nodes.append(open3d.registration.PoseGraphNode(np.linalg.inv(pose)))
        pose_graph.edges.append(open3d.registration.PoseGraphEdge(i, i + 1, transformation, information, uncertain=False))
    
    # Perform registration with all other fragments
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if not (j == i + 1):
                edges = list(pose_graph.edges)
                pose_init = np.identity(4)
                for k in range(i, j):
                    trans = [edge.transformation for edge in edges if edge.source_node_id == k and edge.target_node_id == k + 1]
                    pose_init = np.dot(trans[0], pose_init)

                transformation, information = register_fragments(i, j, np.linalg.inv(pose_init))
                pose_graph.edges.append(open3d.registration.PoseGraphEdge(i, j, transformation, information, uncertain=True))
    

    open3d.io.write_pose_graph("unoptimised2.json", pose_graph)
    
    open3d.registration.global_optimization(pose_graph, method, criteria, option)
    open3d.io.write_pose_graph("optimised2.json", pose_graph)
    
    '''
    source = 6
    target = 7

    pcd_source = open3d.io.read_point_cloud(ROOT + files[source])
    pcd_target = open3d.io.read_point_cloud(ROOT + files[target])

    pcd_source = open3d.geometry.PointCloud.voxel_down_sample(pcd_source, 0.005)
    pcd_target = open3d.geometry.PointCloud.voxel_down_sample(pcd_target, 0.005)

    open3d.geometry.PointCloud.estimate_normals(pcd_source, open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=300))
    pcd_source.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    open3d.geometry.PointCloud.estimate_normals(pcd_target, open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=300))
    pcd_target.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    #open3d.visualization.draw_geometries([pcd_source])
    #open3d.visualization.draw_geometries([pcd_target])

    pose_graph_source = open3d.io.read_pose_graph(POSEGRAPH + files[source][:-4] + ".json")
    pose_graph_target = open3d.io.read_pose_graph(POSEGRAPH + files[target][:-4] + ".json")

    #fpfh_source = open3d.registration.compute_fpfh_feature(pcd_source, open3d.geometry.KDTreeSearchParamHybrid(max_nn=100, radius=0.25))
    #fpfh_target = open3d.registration.compute_fpfh_feature(pcd_target, open3d.geometry.KDTreeSearchParamHybrid(max_nn=100, radius=0.25))

    #res = open3d.registration.registration_fast_based_on_feature_matching(pcd_source, pcd_target, fpfh_source, fpfh_target)
    #res = open3d.registration.registration_icp(pcd_source, pcd_target, 0.05, res.transformation, open3d.registration.TransformationEstimationPointToPlane(), open3d.registration.ICPConvergenceCriteria(max_iteration=500))
    
    res = open3d.registration.registration_icp(pcd_source, pcd_target, 0.05, np.linalg.inv(list(pose_graph_source.nodes)[-1].pose), open3d.registration.TransformationEstimationPointToPlane(), open3d.registration.ICPConvergenceCriteria(max_iteration=500))

    pcd_target.transform(np.linalg.inv(res.transformation))
    #pcd_target.transform(res.transformation)

    open3d.visualization.draw_geometries([pcd_source, pcd_target])
    '''
    pose_graph = open3d.io.read_pose_graph("optimised2.json")
    point_cloud = open3d.geometry.PointCloud()
    for i in range(len(files)):
        pcd = open3d.io.read_point_cloud(ROOT + files[i])
        pcd.transform(pose_graph.nodes[i].pose)
        point_cloud += pcd
        point_cloud = open3d.geometry.PointCloud.voxel_down_sample(point_cloud, 0.008)
    
    open3d.io.write_point_cloud("pcd_final.ply", point_cloud)
    
    open3d.visualization.draw_geometries([point_cloud])
