import cosysairsim as airsim
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3
from builtin_interfaces.msg import Time
import time

class AirSimBridge(Node):
    def __init__(self):
        super().__init__('airsim_bridge')
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.image_pub = self.create_publisher(Image, '/cam0/image_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu0', 10)
        self.br = CvBridge()

        timer_period = 1.0 / 30  # 30Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Image
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        if responses:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            msg = self.br.cv2_to_imgmsg(img_rgb, encoding='rgb8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(msg)

        # IMU
        imu_data = self.client.getImuData()
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.angular_velocity = Vector3(
            x=imu_data.angular_velocity.x_val,
            y=imu_data.angular_velocity.y_val,
            z=imu_data.angular_velocity.z_val
        )
        imu_msg.linear_acceleration = Vector3(
            x=imu_data.linear_acceleration.x_val,
            y=imu_data.linear_acceleration.y_val,
            z=imu_data.linear_acceleration.z_val
        )
        self.imu_pub.publish(imu_msg)

def main():
    rclpy.init()
    node = AirSimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
