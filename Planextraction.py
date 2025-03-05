import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os 

def load_and_preprocess_xyz(xyz_path):
    # Load the XYZ file
    xyz = o3d.io.read_point_cloud(xyz_path)
    xyz_center = xyz.get_center()
    xyz.translate(-xyz_center)

    # Statistical Outlier Removal
    nn = 10
    std_multiplier = 10
    filtered_xyz, _ = xyz.remove_statistical_outlier(nn, std_multiplier)

    return filtered_xyz

def visualize_point_cloud(point_cloud):
    # Visualization of the point cloud
    o3d.visualization.draw_geometries([point_cloud])

def calculate_density_and_downsample(point_cloud):
    # Calculate point cloud density
    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    volume = bounding_box.volume()
    point_cloud_density = len(point_cloud.points) / volume

    # Adjust voxel size based on density
    voxel_size = 3.0 / point_cloud_density  # Adjusted voxel size for less dense data
    downsampled_xyz = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    return downsampled_xyz

def estimate_normals(downsampled_xyz):
    # Normal Estimation
    nn_distance = np.mean(downsampled_xyz.compute_nearest_neighbor_distance())
    radius_normals = nn_distance * 4
    downsampled_xyz.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16),
        fast_normal_computation=True
    )

    return downsampled_xyz

def segment_horizontal_surfaces(downsampled_xyz):
    # Plane Segmentation for Horizontal Surfaces
    pt_to_plane_dist_horizontal = 0.06  # Adjusted for horizontal surfaces
    plane_model_horizontal, inliers_horizontal = downsampled_xyz.segment_plane(distance_threshold=pt_to_plane_dist_horizontal, ransac_n=3, num_iterations=1000)

    # Colorize the inliers and outliers for horizontal surfaces
    inlier_cloud_horizontal = downsampled_xyz.select_by_index(inliers_horizontal)
    outlier_cloud_horizontal = downsampled_xyz.select_by_index(inliers_horizontal, invert=True)
    inlier_cloud_horizontal.paint_uniform_color([1.0, 0, 0])
    outlier_cloud_horizontal.paint_uniform_color([0.6, 0.6, 0.6])

    return inlier_cloud_horizontal, outlier_cloud_horizontal

def segment_vertical_and_remaining_planes(downsampled_xyz):
    # RANSAC for Vertical and Remaining Planes
    pt_to_plane_dist_vertical = 0.06  # Adjusted for vertical walls
    segments = []

    while True:
        _, inliers = downsampled_xyz.segment_plane(distance_threshold=pt_to_plane_dist_vertical, ransac_n=3, num_iterations=1500)
        if len(inliers) < 1000:  # Adjust this threshold based on your data
            break  # Break the loop if no significant plane is found
        segment = downsampled_xyz.select_by_index(inliers)
        segments.append(segment)
        downsampled_xyz = downsampled_xyz.select_by_index(inliers, invert=True)

    return segments

def combine_and_visualize_point_clouds(inlier_horizontal, outlier_horizontal, segments):
    # Combine the segmented horizontal and vertical planes
    combined_point_cloud = inlier_horizontal + o3d.geometry.PointCloud()
    for segment in segments:
        combined_point_cloud += segment

    # Visualize the combined point cloud
    o3d.visualization.draw_geometries([combined_point_cloud])

if __name__ == "__main__":
    # Specify the path to the XYZ file
    xyz_path = r".../lod2_pc_without_veg_1.xyz"

    # Load and preprocess the point cloud
    filtered_point_cloud = load_and_preprocess_xyz(xyz_path)

    # Visualize the filtered point cloud
    visualize_point_cloud(filtered_point_cloud)

    # Calculate density, downsample, and estimate normals
    downsampled_point_cloud = calculate_density_and_downsample(filtered_point_cloud)
    downsampled_point_cloud = estimate_normals(downsampled_point_cloud)

    # Segment horizontal surfaces
    inliers_horizontal, outlier_horizontal = segment_horizontal_surfaces(downsampled_point_cloud)

    # Visualize the results for horizontal surfaces
    visualize_point_cloud(inliers_horizontal + outlier_horizontal)

    # Segment vertical and remaining planes
    segments = segment_vertical_and_remaining_planes(downsampled_point_cloud)

    # Combine and visualize point clouds
    combine_and_visualize_point_clouds(inliers_horizontal, outlier_horizontal, segments)

def save_point_cloud_as_xyz(point_cloud, filename):
    # Save point cloud as XYZ file
    points = np.asarray(point_cloud.points)
    np.savetxt(filename, points, fmt='%f %f %f')

def combine_and_save_point_clouds_as_xyz(inlier_horizontal, segments, save_folder):
    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Save the segmented planes as XYZ files
    inlier_horizontal_path = os.path.join(save_folder, "horizontal_plane.xyz")
    save_point_cloud_as_xyz(inlier_horizontal, inlier_horizontal_path)

    for i, segment in enumerate(segments):
        segment_path = os.path.join(save_folder, f"vertical_plane_{i}.xyz")
        save_point_cloud_as_xyz(segment, segment_path)

# ...

if __name__ == "__main__":

    # Specify the folder to save the segmented planes
    save_folder = r".../Plane2Plane writers/CityGML pointcloud"

    # Combine and save point clouds as XYZ files
    combine_and_save_point_clouds_as_xyz(inliers_horizontal, segments, save_folder)
