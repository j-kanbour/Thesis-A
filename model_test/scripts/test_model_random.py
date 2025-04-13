#!/usr/bin/env python3

import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np


def random_rotate_translate_pointcloud(points):
    # Random rotation (Euler angles)
    angles = np.random.uniform(0, 2*np.pi, size=3)  # roll, pitch, yaw
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]),  np.cos(angles[0])]
    ])
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]),  np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    # Random translation in +/- 1 meter in each direction
    t = np.random.uniform(-1, 1, size=3)

    # Apply rotation and translation
    transformed = (points @ R.T) + t

    return transformed, t, R


def read_ply():
    file_path = "/home/toyota/catkin_ws/src/image_player/models/obj_000001.ply"
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Set all points to pink (RGB = [1.0, 0.0, 1.0])
    pcd.paint_uniform_color([1.0, 0.0, 1.0])  # Pink color
    
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def publish_pointcloud(points, colors, publisher):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1)  # Adding RGB field
    ]

    # Convert color from RGB (0-1 range) to uint32 (0-16777215)
    rgb_values = [(int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255) for r, g, b in colors]
    
    cloud_data = [tuple(pt) + (rgb,) for pt, rgb in zip(points, rgb_values)]
    cloud_msg = pc2.create_cloud(header, fields, cloud_data)
    publisher.publish(cloud_msg)


def publish_centroid(points, pub_pose):
    centroid = np.mean(points, axis=0)
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "map"
    pose.pose.position.x = centroid[0]
    pose.pose.position.y = centroid[1]
    pose.pose.position.z = centroid[2]
    pose.pose.orientation.w = 1.0  # No orientation needed
    pub_pose.publish(pose)


def main():
    rospy.init_node('ply_pointcloud_publisher', anonymous=True)
    pub_cloud = rospy.Publisher("/ply_pointcloud", PointCloud2, queue_size=1)
    pub_centroid = rospy.Publisher("/ply_centroid", PoseStamped, queue_size=1)
    rate = rospy.Rate(0.5)  # 1 Hz

    rospy.loginfo("Reading PLY file...")
    try:
        base_points, base_colors = read_ply()
        rospy.loginfo("Loaded base point cloud with {} points.".format(len(base_points)))
    except Exception as e:
        rospy.logerr("Failed to load .ply file: {}".format(e))
        return

    while not rospy.is_shutdown():
        transformed_points, t, R = random_rotate_translate_pointcloud(base_points)
        publish_pointcloud(transformed_points, base_colors, pub_cloud)
        publish_centroid(transformed_points, pub_centroid)
        rospy.loginfo("Published transformed point cloud and centroid.")
        rate.sleep()


if __name__ == "__main__":
    main()
