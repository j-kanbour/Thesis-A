#!/usr/bin/env python3

import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct

def random_rotate_pointcloud(points):
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
    return points @ R.T

def read_ply():
    file_path = "/home/toyota/catkin_ws/src/image_player/models/obj_000001.ply"
    pcd = o3d.io.read_point_cloud(file_path)

    # Set all points to pink (RGB = [1.0, 0.0, 1.0])
    pcd.paint_uniform_color([1.0, 0.0, 1.0])

    return np.asarray(pcd.points), np.asarray(pcd.colors)

def publish_pointcloud(points, colors, publisher):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.FLOAT32, 1),
    ]

    cloud_data = []
    for pt, color in zip(points, colors):
        # Convert RGB [0.0, 1.0] to 24-bit packed float
        r, g, b = (np.array(color) * 255).astype(np.uint8)
        rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        cloud_data.append((pt[0], pt[1], pt[2], rgb))

    cloud_msg = pc2.create_cloud(header, fields, cloud_data)
    publisher.publish(cloud_msg)

def remove_half_by_axis(points, colors):
    axis = np.random.choice([0, 1, 2])
    remove_condition = np.random.choice([True, False])

    if remove_condition:
        mask = points[:, axis] < 0
    else:
        mask = points[:, axis] >= 0

    return points[mask], colors[mask]

def main():
    rospy.init_node('ply_pointcloud_publisher', anonymous=True)
    pub = rospy.Publisher("/ply_pointcloud", PointCloud2, queue_size=1)
    rate = rospy.Rate(0.5)  # 1 Hz

    rospy.loginfo("Reading PLY file...")
    try:
        points, colors = read_ply()
        points = random_rotate_pointcloud(points)
        rospy.loginfo("Loaded and rotated point cloud with {} points.".format(len(points)))
    except Exception as e:
        rospy.logerr("Failed to load or process .ply file: {}".format(e))
        return

    while not rospy.is_shutdown():
        partial_points, partial_colors = remove_half_by_axis(points, colors)
        rospy.loginfo("Publishing partial point cloud with {} points.".format(len(partial_points)))
        publish_pointcloud(partial_points, partial_colors, pub)
        rate.sleep()

if __name__ == "__main__":
    main()
