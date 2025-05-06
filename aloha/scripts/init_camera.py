from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
    ImageRecorder
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy
import cv2
import time
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

class AprilTagListener:
    def __init__(self, node: Node):
        self.node = node
        self.transform_dict = {}
        self.image_dict = {}
        
        self.bridge = CvBridge()
        self.camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        
        for camera_name in self.camera_names:
            node.create_subscription(
                Image,
                f"/{camera_name}/camera/color/image_rect_raw",
                lambda msg, cam_name=camera_name: self.image_callback(cam_name, msg),
                10
            )
            
            node.create_subscription(
                TFMessage,
                f"/{camera_name}/tf",
                lambda msg, cam_name=camera_name: self.listener_callback(cam_name, msg),
                10
            )

                
    def image_callback(self, cam_name: str, data: Image):
        self.image_dict[cam_name] = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
    def listener_callback(self, cam_name: str, data: TFMessage):
        if len(data.transforms) == 0:
            return
        self.transform_dict[cam_name] = {}
        for transform in data.transforms:
            transform: TransformStamped 
            self.transform_dict[cam_name][transform.child_frame_id] = transform.transform
        

def main():
    node = create_interbotix_global_node('aloha')
    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=True,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=True,
    )
    leader_bot_left = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='leader_left',
        node=node,
        iterative_update_fk=True,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model='wx250s',
        robot_name='leader_right',
        node=node,
        iterative_update_fk=True,
    )

    robot_startup(node)
    torque_off(follower_bot_left)
    torque_off(follower_bot_right)
    torque_off(leader_bot_left)
    torque_off(leader_bot_right)
        
    apriltag_listener = AprilTagListener(node)
    time.sleep(3)
    while len(apriltag_listener.image_dict) < 3 or len(apriltag_listener.transform_dict) < 2:
        node.get_logger().info("Waiting for images and transforms...")
        time.sleep(1)
    
    image_dict = apriltag_listener.image_dict
    transform_dict = apriltag_listener.transform_dict
    # print(transform_dict)
    
    print(f"{len(image_dict)} images received")
    print(f"{len(transform_dict)} transforms received")
    
    for camera_name, image in image_dict.items():
        print(f"Camera: {camera_name}")
        print(f"RGB Image size: {image.shape}")
        cv2.imwrite(f"{camera_name}_rgb.png", image)
        
    for transform_name, transform in transform_dict.items():
        print(f"Transform: {transform_name}")
        print(f"Transform data for {transform_name}:")
        print(transform)
        
    ee_pose = follower_bot_left.arm.get_ee_pose()
    
    import ipdb; ipdb.set_trace() # fmt: skip
        
    robot_shutdown(node)

if __name__ == "__main__":
    main()
