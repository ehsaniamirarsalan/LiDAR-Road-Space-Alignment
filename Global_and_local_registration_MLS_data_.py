import numpy as np
import open3d as o3d
import copy

# Function to draw registration result(parameters: source and target point cloud,  transformation parameters and output window name)
def draw_registration_result(source, target, transformation, window):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Set color for source point cloud
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Set color for target point cloud
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window, width=2500, height=2500, left=50, top=50)

# Function to preprocess point cloud( preprocess point cloud for voxel downsampling, normal estimation and FPFH generation)
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# Function to prepare dataset
def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud("Provide the file path for the source pointc cloud")
    target = o3d.io.read_point_cloud("Provide the file path for the target(reference) point cloud")

    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4), 'original point cloud')

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# Specify voxel size
voxel_size = .75  # means 75mm for this dataset

# Prepare dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

# Function to execute global registration
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# Execute global registration
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_ransac, result_ransac.transformation)
draw_registration_result(source, target, result_ransac.transformation, 'Global registration')

# Specify threshold for point-to-point ICP
threshold = voxel_size * 10
#adjust threshold based on point cloud(based on distances between corresponding point clouds)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, result_ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
print("")

draw_registration_result(source, target, reg_p2p.transformation, 'point to point registration')

# Specify radius for normal estimation in point-to-plane ICP
radius_normal = voxel_size * .5
print(":: Estimate normal with search radius %.3f." % radius_normal)
target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30
