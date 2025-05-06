#!/usr/bin/env python3

# ros2 run aloha_camera single_stream_node --ros-args -p reference_path:=./reference.png -r __ns:=/cam_left_wrist

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class BlendNode(Node):
    def __init__(self):
        super().__init__('single_stream_node')

        # Declare parameters
        self.declare_parameter('camera_topic', 'camera/color/image_rect_raw')
        self.declare_parameter('reference_path', '')
        self.declare_parameter('alpha', 0.5)
        self.declare_parameter('beta', 0.5)
        self.declare_parameter('window_name', 'Blended View')

        # Get parameters
        camera_topic = self.get_parameter('camera_topic').value
        self.ref_path = self.get_parameter('reference_path').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.window = self.get_parameter('window_name').value

        # Attempt to load reference if it exists; otherwise start with None
        if self.ref_path and os.path.isfile(self.ref_path):
            self.reference = cv2.imread(self.ref_path, cv2.IMREAD_COLOR)
            if self.reference is None:
                self.get_logger().warning(f"Could not read reference image at '{self.ref_path}'. Starting without reference.")
        else:
            self.reference = None
            if not self.ref_path:
                self.get_logger().info("No reference_path provided—will display live stream only until 's' is pressed.")
            else:
                self.get_logger().info(f"Reference image '{self.ref_path}' not found—will display live stream only until 's' is pressed.")

        # Bridge and subscriber
        self.bridge = CvBridge()
        self.create_subscription(Image, camera_topic, self.image_callback, 10)

        # OpenCV window
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.get_logger().info(f"BlendNode initialized on topic '{camera_topic}'. Press 's' to save reference.")

    def image_callback(self, msg: Image):
        # Convert to OpenCV
        try:
            live_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Decide what to show: blended if reference exists, else raw
        if self.reference is not None:
            # match sizes
            if live_img.shape[:2] != self.reference.shape[:2]:
                ref_resized = cv2.resize(self.reference, (live_img.shape[1], live_img.shape[0]))
            else:
                ref_resized = self.reference

            display = cv2.addWeighted(live_img, self.alpha, ref_resized, self.beta, 0)
        else:
            display = live_img

        # Show and poll key
        cv2.imshow(self.window, display)
        key = cv2.waitKey(1) & 0xFF

        # If user pressed 's', save live frame as new reference
        if key == ord('s') and self.ref_path and self.ref_path != '':
            try:
                cv2.imwrite(self.ref_path, live_img)
                self.reference = live_img.copy()
                self.get_logger().info(f"Saved new reference image to '{self.ref_path}'")
            except Exception as e:
                self.get_logger().error(f"Failed to save reference image: {e}")
        elif key == ord('q'):
            self.get_logger().info("Exiting...")
            raise KeyboardInterrupt

def main(args=None):
    rclpy.init(args=args)
    node = BlendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()