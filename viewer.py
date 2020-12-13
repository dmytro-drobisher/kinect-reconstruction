##################################
# Reads and displays point cloud #
##################################

import sys
import open3d

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)
pcd = open3d.io.read_point_cloud(sys.argv[1])
open3d.visualization.draw_geometries([pcd])
