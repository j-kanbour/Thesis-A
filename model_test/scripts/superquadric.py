#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
import ros_numpy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import time

def ros_to_o3d(ros_cloud):
    pc_np = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)
    xyz = np.zeros((pc_np.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = pc_np['x']
    xyz[:, 1] = pc_np['y']
    xyz[:, 2] = pc_np['z']
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def o3d_to_ros(pcd, frame_id="map"):
    points = np.asarray(pcd.points)
    pc_struct = np.zeros(points.shape[0], dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32)
    ])
    pc_struct['x'] = points[:, 0]
    pc_struct['y'] = points[:, 1]
    pc_struct['z'] = points[:, 2]

    ros_cloud = ros_numpy.point_cloud2.array_to_pointcloud2(pc_struct)
    ros_cloud.header = Header()
    ros_cloud.header.stamp = rospy.Time.now()
    ros_cloud.header.frame_id = frame_id
    return ros_cloud

def fit_pca_ellipsoid(pcd):
    # Step 1: Denoise
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Step 2: PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Step 3: Transform to PCA-aligned space
    aligned_points = centered @ eigvecs
    min_bound = np.min(aligned_points, axis=0)
    max_bound = np.max(aligned_points, axis=0)
    radii = (max_bound - min_bound) / 2.0 * 1.05  # Slight overscale

    # Step 4: Sample ellipsoid in PCA-aligned space
    u = np.linspace(0, np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)
    x = radii[0] * np.sin(u) * np.cos(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(u)

    ellipsoid = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Step 5: Transform back to original space
    rotated = ellipsoid @ eigvecs.T
    translated = rotated + centroid

    ellipsoid_pcd = o3d.geometry.PointCloud()
    ellipsoid_pcd.points = o3d.utility.Vector3dVector(translated)

    return ellipsoid_pcd

class SuperquadricNode:
    def __init__(self):
        rospy.init_node('superquadric_fitter_node', anonymous=True)
        self.sub = rospy.Subscriber('/ply_pointcloud', PointCloud2, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/superquadric_pointcloud', PointCloud2, queue_size=1)

        self.time_track = []

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("Improved PCA Superquadric Node Started")
    
    def cleanup(self):
        if self.time_track:
            print(f"\n\n=== Superquadric Summary ===")
            print(f"Total runs: {len(self.time_track)}")
            print(f"Average time: {np.mean(self.time_track):.4f}s")
        else:
            print("No alignment was performed.")

    def callback(self, ros_cloud):
        try:
            pcd = ros_to_o3d(ros_cloud)
            if len(pcd.points) < 50:
                rospy.logwarn("Not enough points for fitting.")
                return

            start_time = time.time()

            superquadric_pcd = fit_pca_ellipsoid(pcd)

            elapsed = time.time() - start_time
            self.time_track.append(elapsed)

            ros_msg = o3d_to_ros(superquadric_pcd, frame_id=ros_cloud.header.frame_id)
            self.pub.publish(ros_msg)
        except Exception as e:
            rospy.logerr(f"Error in fitting: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = SuperquadricNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
