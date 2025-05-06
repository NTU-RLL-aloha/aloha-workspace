from typing import List, Tuple

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from tf_transformations import quaternion_matrix

import numpy as np
from numpy.linalg import eigh


class Localization(Node):
    def __init__(self):
        super().__init__("apriltag_localization_node")
        self.declare_parameter("family", "")
        self.declare_parameter("offset_ids", [0])
        self.declare_parameter("origin_topic", "/aloha/apriltag")
        self.declare_parameter("origin_frame_id", "apriltag_origin")

        self.tag_family = self.get_parameter("family").get_parameter_value().string_value
        self.offset_ids = self.get_parameter("offset_ids").get_parameter_value().integer_array_value
        self.origin_topic = self.get_parameter("origin_topic").get_parameter_value().string_value
        self.origin_frame_id = self.get_parameter("origin_frame_id").get_parameter_value().string_value

        self.offsets = {}
        self.frame_ids_to_ids = {}
        for id in self.offset_ids:
            self.declare_parameter(f"offset_{id}", [0.0, 0.0, 0.0])

            frame_id = f"tag{self.tag_family}:{id}"
            offset = (
                self.get_parameter(f"offset_{id}")
                .get_parameter_value()
                .double_array_value
            )

            self.frame_ids_to_ids[frame_id] = id
            self.offsets[id] = np.array(offset)

        self.get_logger().info(f"Tag family: {self.tag_family}")
        self.get_logger().info(f"Frame IDs: {self.frame_ids_to_ids}")
        self.get_logger().info(f"Offsets: {self.offsets}")

        self.subscription = self.create_subscription(
            TFMessage, "/tf", self.listener_callback, 10
        )
        self.tf_publisher = self.create_publisher(
            TFMessage, "/tf", 10
        )
        self.dedicated_publisher = self.create_publisher(
            TFMessage, self.origin_topic, 10
        )

    def filter_transform(self, data: TFMessage) -> List[Tuple[int, TransformStamped]]:
        filtered_id_transforms = []
        for transform in data.transforms:
            if transform.child_frame_id in self.frame_ids_to_ids.keys():
                id = self.frame_ids_to_ids[transform.child_frame_id]
                filtered_id_transforms.append((id, transform))

        return filtered_id_transforms
    
    @staticmethod
    def transform_offset(offset: np.ndarray, transform: TransformStamped) -> np.ndarray:
        quat = transform.transform.rotation
        trans = transform.transform.translation
        # Rotate the offset to the tag frame
        Q = [quat.x, quat.y, quat.z, quat.w]
        M = quaternion_matrix(Q)
        offset_homog = np.concatenate((offset, [0]))
        offset_rot = M.dot(offset_homog)[:3]
        # Translate the offset to the camera frame
        T = np.array([trans.x, trans.y, trans.z])
        offset_trans = T + offset_rot

        return offset_trans
    
    @staticmethod
    def average_quaternions(Q_list: List[Quaternion], weights=None) -> Quaternion:
        """
        Q_list: list of quaternions
        weights: optional list of scalar weights
        """
        M = len(Q_list)
        if weights is None:
            weights = [1.0]*M
        # Build accumulator
        A = np.zeros((4,4))
        wsum = 0.0
        for Q, w in zip(Q_list, weights):
            q = np.array([Q.x, Q.y, Q.z, Q.w])
            v = np.array(q)
            A += w * np.outer(v, v)
            wsum += w
        A /= wsum
        # Principal eigenvector
        eigvals, eigvecs = eigh(A)
        avg_q = eigvecs[:, np.argmax(eigvals)]
        # Return normalized quaternion
        avg_q /= np.linalg.norm(avg_q)
        
        return Quaternion(
            x=avg_q[0],
            y=avg_q[1],
            z=avg_q[2],
            w=avg_q[3],
        )

    def listener_callback(self, data: TFMessage):
        id_transforms = self.filter_transform(data)
        if len(id_transforms) == 0:
            return

        origin_array = np.zeros(3)
        origin_rotations = []
        for id, transform in id_transforms:
            offset = self.offsets[id]
            origin_rotation = transform.transform.rotation
            transformed_offset = self.transform_offset(offset, transform)

            origin_array += transformed_offset / len(id_transforms)
            origin_rotations.append(origin_rotation)

            self.get_logger().debug(f"Transformed origin for ID {id}: {transformed_offset}")
         
        origin_translation = Vector3(
            x=origin_array[0],
            y=origin_array[1],
            z=origin_array[2],
        )
        origin_rotation = self.average_quaternions(origin_rotations)
        
        tf_message = TFMessage(
            transforms=[
                TransformStamped(
                    header=data.transforms[0].header,
                    child_frame_id=self.origin_frame_id,
                    transform=Transform(
                        translation=origin_translation,
                        rotation=origin_rotation,
                    ),
                )
            ]
        )
        self.tf_publisher.publish(tf_message)
        self.dedicated_publisher.publish(tf_message)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = Localization()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
