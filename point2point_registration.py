import open3d as o3d
import numpy as np
import pandas as pd
import os
import copy 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_users_param ():
    target_path = input("Enter the file path to the target pointc cloud: ")
    source_path = input("Enter the file path to the folder containing submaps: ")
    threshold = int(input("Enter the threshold value (recommended value - 10): "))

    return target_path, source_path, threshold

def calculate_trans_matrix (source, target, threshold, trans_init):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        threshold, 
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500, 
                                                          relative_fitness=1e-06, 
                                                          relative_rmse=1e-06))
    transformation = reg_p2p.transformation
    return (transformation, reg_p2p)

def get_pc_files (root_folder):
    xyz_files = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('_world.xyz'):
                file_path = os.path.join(root, file)
                xyz_files.append(file_path)

    return xyz_files

def write_accuracy_results(registration_results, evaluation,i, state):
    registration_results.append({'Submap Number': i, 'State': state,'Fitness': evaluation.fitness, 'Inlier RMSE': evaluation.inlier_rmse})

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def screenshot(source, target, save_path):
    min_bound = np.min(np.asarray(source.points), axis=0)
    max_bound = np.max(np.asarray(source.points), axis=0)
    screenshots_folder = "output/screenshots"
    if not os.path.exists(screenshots_folder):
        os.makedirs(screenshots_folder)
    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=2500)
    vis.add_geometry(source)
    vis.add_geometry(target)
    vis.poll_events()
    ctr = vis.get_view_control()
    ctr.set_lookat((min_bound + max_bound) / 2.0)
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(10)
    vis.capture_screen_image(save_path)
    vis.destroy_window()

    print(f"Screenshot saved: {save_path}")

def save_results_to_pdf(submap_number, 
                        registration_method, 
                        before_registration_path, 
                        after_registration_path, 
                        before_fitness,
                        before_RMSE,
                        after_fitness, 
                        after_RMSE):
    results_folder = "output/results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    pdf_filename = f"Results_submap_{submap_number}_p2p.pdf"
    pdf_path = os.path.join(results_folder, pdf_filename)

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(8.27, 11.69)) 
        plt.suptitle(f"Submap {submap_number} - {registration_method}",  fontsize=30, fontweight='bold', ha='center', fontname='Arial')

        before_image = plt.imread(before_registration_path)
        axes[0].imshow(before_image)
        axes[0].axis("off")
        axes[0].set_title("Before Registration", fontname='Arial')

        axes[0].text(0.5, -0.15, f"Fitness: {before_fitness:.2f}\nInlier RMSE: {before_RMSE:.2f}", 
                    fontsize=12, ha="center", va="center", fontname='Arial', transform=axes[0].transAxes)
        axes[0].axis("off")

        after_image = plt.imread(after_registration_path)
        axes[1].imshow(after_image)
        axes[1].axis("off")
        axes[1].set_title("After Registration", fontname='Arial')

        axes[1].text(0.5, -0.15, f"Fitness: {after_fitness:.2f}\nInlier RMSE: {after_RMSE:.2f}", 
                    fontsize=12, ha="center", va="center", fontname='Arial', transform=axes[1].transAxes)
        axes[1].axis("off")

        plt.tight_layout()

        pdf.savefig()
        plt.close()

    print(f"PDF saved: {pdf_path}")

def plot_results(registration_results):
    df = pd.DataFrame(registration_results)

    df_before = df[df['State'] == 'before']
    df_after = df[df['State'] == 'after']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    width = 0.35
    bar_positions_before = np.arange(len(df_before['Submap Number']))
    bar_positions_after = bar_positions_before + width

    axes[0].bar(bar_positions_before, df_before['Inlier RMSE'], width, label='Before', color='lightcoral')
    axes[0].bar(bar_positions_after, df_after['Inlier RMSE'], width, label='After', color='skyblue')

    axes[0].set_title('Inlier RMSE Comparison (Before vs. After)')
    axes[0].set_xlabel('Submap Number')
    axes[0].set_ylabel('Inlier RMSE')
    axes[0].set_xticks(bar_positions_before + width / 2)
    axes[0].set_xticklabels(df_before['Submap Number'])
    axes[0].legend()

    df_difference = pd.DataFrame({
    'Submap Number': df_before['Submap Number'],
    'Inlier RMSE Difference': df_after['Inlier RMSE'].values - df_before['Inlier RMSE'].values
    })

    axes[1].bar(df_difference['Submap Number'], df_difference['Inlier RMSE Difference'], color='green')
    axes[1].set_title('Inlier RMSE Difference (After - Before)')
    axes[1].set_xlabel('Submap Number')
    axes[1].set_ylabel('Inlier RMSE Difference')

    plt.tight_layout()

    results_folder = "output/results"
    png_filename = "Inlier_RMSE_Comparison.png"
    png_path = os.path.join(results_folder, png_filename)
    plt.savefig(png_path)

    plt.show()

    print(f"Plot saved as: {png_path}")

def save_pc(point_cloud):
    points = np.asarray(point_cloud.points)
    header_line = "x y z\n"
    points_str = "\n".join([" ".join(map(str, p)) for p in points])
    final_point_cloud_str = header_line + points_str
    output_path = f'output/submaps/submap_{i}_world_p2p.xyz'
    with open(output_path, "w") as file:
        file.write(final_point_cloud_str)

    print(f"Transformed point cloud saved to {output_path}")

def tr_and_rt_param(trans_matrix):
    translation = trans_matrix[:3, 3]
    length = np.linalg.norm(translation)
    rotation_matrix = trans_matrix[:3, :3]
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = -np.arcsin(rotation_matrix[2, 0])
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return length, roll, pitch, yaw

def save_length_and_roll(submap_number, length, roll):
    output_folder = "output"
    output_path = f"{output_folder}/length_and_roll_results.txt"

    if not os.path.exists(output_path):
        with open(output_path, "w") as file:
            file.write("Submap Number\tLength\tRoll\n")

    with open(output_path, "a") as file:
        file.write(f"{submap_number}\t{length}\t{np.degrees(roll)}\n")

    print(f"Length and Roll results saved to {output_path}")

if __name__ == "__main__":

    target_path, source_path, threshold = get_users_param()

    registration_results = []

    source_pcs= get_pc_files(source_path)
    source_pcs.sort()

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    matrix_folder = "output/transformation_matrix"
    if not os.path.exists(matrix_folder):
        os.makedirs(matrix_folder)

    submap_folder = "output/submaps"
    if not os.path.exists(submap_folder):
        os.makedirs(submap_folder)

    for i in range(0, len(source_pcs)):
        target = o3d.io.read_point_cloud(target_path)
        source = o3d.io.read_point_cloud(source_pcs[i])
        
        screenshot_path_before = f"output/screenshots/submap_{i}_{threshold}_before.png"
        screenshot(source, target, screenshot_path_before)

        trans_init = np.identity(4)
        accuracy_before = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)

        write_accuracy_results(registration_results, accuracy_before, i, "before")

        trans_matrix, accuracy_after = calculate_trans_matrix (source, target, threshold, trans_init)
        
        write_accuracy_results(registration_results, accuracy_after, i, "after")
    
        file_path = f"output/transformation_matrix/output_{i}_{threshold}.txt"
        np.savetxt(file_path, trans_matrix, fmt='%f', delimiter='\t')

        length, roll, pitch, yaw = tr_and_rt_param(trans_matrix)
        save_length_and_roll(i, length, roll)

        print("Length of the translation vector:", length)
        print("Roll:", np.degrees(roll))
    
        result = source.transform(trans_matrix)
        save_pc(result)

        screenshot_path_after = f"output/screenshots/submap_{i}_{threshold}_after.png"
        screenshot(result, target, screenshot_path_after)

        save_results_to_pdf(str(i), f"\nIterative closest point (ICP)\n Point to point registration", 
                            screenshot_path_before, 
                            screenshot_path_after, 
                            accuracy_before.fitness,
                            accuracy_before.inlier_rmse,
                            accuracy_after.fitness,
                            accuracy_after.inlier_rmse)

    results_df = pd.DataFrame(registration_results)
    print("Registration Results Table:")
    print(results_df)

    output_file_path = "output/registration_results.txt"
    results_df.to_csv(output_file_path, sep='\t', index=False)

    print(f"Registration results saved to {output_file_path}")

    plot_results(registration_results)
