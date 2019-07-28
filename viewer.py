##################################
# Reads and displays point cloud #
##################################

import sys
import open3d

pcd = open3d.read_point_cloud(sys.argv[1])
open3d.draw_geometries([pcd])