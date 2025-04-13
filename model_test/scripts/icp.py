#!/usr/bin/env python3

import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import numpy as np
import ros_numpy
import time
import std_msgs.msg
from scipy.spatial.transform import Rotation as R


def ros_to_open3d(ros_cloud):
    pc_np = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)
    xyz = np.zeros((pc_np.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = pc_np['x']
    xyz[:, 1] = pc_np['y']
    xyz[:, 2] = pc_np['z']
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)
    return o3d_pc


def open3d_to_ros(pcd, frame_id="map"):
    points = np.asarray(pcd.points)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    return pc2.create_cloud_xyz32(header, points)


def downsample_point_cloud(pcd, voxel_size=0.01):
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = downsample_point_cloud(pcd, voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
    )
    return pcd_down, fpfh


def fallback_translation_only(source, target):
    src_center = source.get_center()
    tgt_center = target.get_center()
    T = np.eye(4)
    T[:3, 3] = tgt_center - src_center
    return T


def get_initial_transform(source, target, voxel_size=0.05):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
        )
        if result.transformation.trace() == 4.0:
            rospy.logwarn("RANSAC returned identity. Using fallback.")
            return fallback_translation_only(source, target)
        return result.transformation
    except Exception as e:
        rospy.logwarn(f"Initial alignment failed: {e}")
        return fallback_translation_only(source, target)


def matrix_to_pose(T):
    translation = T[:3, 3]
    rotation = R.from_matrix(T[:3, :3])
    roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
    return translation[0], translation[1], translation[2], roll, pitch, yaw


class ICPAligner:
    def __init__(self):
        rospy.init_node("icp_aligner_node", anonymous=True)

        self.source_ply_path = "/home/toyota/catkin_ws/src/image_player/models/obj_000021.ply"
        self.source_pcd = o3d.io.read_point_cloud(self.source_ply_path)
        self.source_pcd = downsample_point_cloud(self.source_pcd, voxel_size=0.01)

        self.frame_id = "map"
        self.pub = rospy.Publisher("/aligned_pointcloud", PointCloud2, queue_size=1)
        rospy.Subscriber("/ply_pointcloud", PointCloud2, self.callback)

        self.time_track = []
        self.fitness = []
        self.rmse = []

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("ICP Aligner Node is ready.")
        rospy.spin()

    def cleanup(self):
        if self.time_track:
            print(f"\n\n=== ICP Summary ===")
            print(f"Total runs: {len(self.time_track)}")
            print(f"Average time: {np.mean(self.time_track):.4f}s")
            print(f"Average fitness: {np.mean(self.fitness):.4f}")
            print(f"Average RMSE: {np.mean(self.rmse):.4f}\n")
        else:
            print("No alignment was performed.")

    def callback(self, ros_cloud):
        target_pcd = ros_to_open3d(ros_cloud)
        if target_pcd is None or len(target_pcd.points) == 0:
            return

        target_pcd = downsample_point_cloud(target_pcd, voxel_size=0.01)
        source_copy = o3d.geometry.PointCloud(self.source_pcd)  # avoid in-place transform

        start_time = time.time()

        #init_transform = get_initial_transform(source_copy, target_pcd)
        #x, y, z, roll, pitch, yaw = matrix_to_pose(init_transform)
        #rospy.loginfo(f"Initial Guess - Pos: ({x:.2f}, {y:.2f}, {z:.2f}) | Rot: ({roll:.2f}, {pitch:.2f}, {yaw:.2f})")

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_copy, target_pcd,
            max_correspondence_distance=100,
            #init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        final_transform = reg_p2p.transformation
        aligned_pcd = source_copy.transform(final_transform)

        evaluation = o3d.pipelines.registration.evaluate_registration(
            aligned_pcd, target_pcd, max_correspondence_distance=0.05)

        ros_aligned = open3d_to_ros(aligned_pcd, self.frame_id)
        self.pub.publish(ros_aligned)

        elapsed = time.time() - start_time
        self.time_track.append(elapsed)
        self.fitness.append(evaluation.fitness)
        self.rmse.append(evaluation.inlier_rmse)

        rospy.loginfo(f"ICP Complete | Time: {elapsed:.3f}s | Fitness: {evaluation.fitness:.4f} | RMSE: {evaluation.inlier_rmse:.4f}")

        # Optional: Visual debug
        # o3d.visualization.draw_geometries([
        #     aligned_pcd.paint_uniform_color([0, 1, 0]),
        #     target_pcd.paint_uniform_color([1, 0, 0])
        # ])


if __name__ == "__main__":
    try:
        ICPAligner()
    except rospy.ROSInterruptException:
        pass
