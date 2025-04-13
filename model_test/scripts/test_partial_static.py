#!/usr/bin/env python3

import rospy
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

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
    file_path = "/home/toyota/catkin_ws/src/image_player/models/obj_000021.ply"
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Set all points to pink (RGB = [1.0, 0.0, 1.0])
    pcd.paint_uniform_color([1.0, 0.0, 1.0])  # Pink color
    
    return np.asarray(pcd.points), np.asarray(pcd.colors)

def publish_pointcloud(points, publisher):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    cloud_data = [tuple(pt) for pt in points]
    cloud_msg = pc2.create_cloud(header, fields, cloud_data)
    publisher.publish(cloud_msg)

def remove_half_by_axis(points):
    # Randomly select an axis (0: x, 1: y, 2: z)
    axis = np.random.choice([0, 1, 2])
    
    # Define the condition to remove half of the points on that axis
    if axis == 0:  # x axis
        # Keep points where x is >= 0
        points = points[points[:, axis] >= 0]
    elif axis == 1:  # y axis
        # Keep points where y is >= 0
        points = points[points[:, axis] >= 0]
    else:  # z axis
        # Keep points where z is >= 0
        points = points[points[:, axis] >= 0]

    return points

def main():
    rospy.init_node('ply_pointcloud_publisher', anonymous=True)
    pub = rospy.Publisher("/ply_pointcloud", PointCloud2, queue_size=1)
    rate = rospy.Rate(0.5)  # 1 Hz

    rospy.loginfo("Reading PLY file...")
    try:
        full_points = read_ply()
        full_points = random_rotate_pointcloud(full_points)

        # Remove half of the points based on random axis
        partial_points = remove_half_by_axis(full_points)
        rospy.loginfo("Publishing partial point cloud with {} points.".format(len(partial_points)))
    except Exception as e:
        rospy.logerr("Failed to load or process .ply file: {}".format(e))
        return

    while not rospy.is_shutdown():
        publish_pointcloud(partial_points, pub)
        rate.sleep()

if __name__ == "__main__":
    main()
