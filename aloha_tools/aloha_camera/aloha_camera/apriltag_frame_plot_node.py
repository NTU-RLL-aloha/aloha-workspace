#!/usr/bin/env python3

# ros2 run aloha_camera apriltag_frame_plot_node --ros-args -r __ns:=/cam_left_wrist -p reference_transform:="./reference.yaml"

import os
import yaml
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Transform
from tf_transformations import quaternion_matrix
import matplotlib.pyplot as plt


class FramePlotNode(Node):
    def __init__(self):
        super().__init__('apriltag_frame_plot_node')

        # Parameters
        self.declare_parameter('reference_path', 'ref_transform.yaml')
        self.declare_parameter('transform_topic', 'tf')
        ref_yaml = self.get_parameter('reference_path').value
        topic    = self.get_parameter('transform_topic').value

        # Path for saving
        self.ref_yaml = ref_yaml

        # Load reference transform if exists
        self.T_ref = None
        if os.path.isfile(ref_yaml):
            try:
                with open(ref_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                rt = data.get('transform', None)
                if rt:
                    self.T_ref = Transform(
                        translation=Transform().translation.__class__(**rt['translation']),
                        rotation=Transform().rotation.__class__(**rt['rotation'])
                    )
                    self.get_logger().info(f"Loaded reference transform from '{ref_yaml}'")
                else:
                    self.get_logger().warning(f"No 'transform' key in '{ref_yaml}', ignoring reference.")
            except Exception as e:
                self.get_logger().warning(f"Failed to load reference YAML '{ref_yaml}': {e}")
        else:
            self.get_logger().info(f"Reference YAML '{ref_yaml}' not found—will ignore until saved.")

        # Storage for last dynamic transform
        self.latest_dyn = None

        # Set up matplotlib 3D plot
        self.fig = plt.figure()
        self.ax  = self.fig.add_subplot(111, projection='3d')
        self._init_plot()

        # Capture keypresses
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Subscribe to namespaced "tf" topic
        self.sub = self.create_subscription(
            TFMessage,
            topic,
            self.tf_callback,
            10
        )
        self.get_logger().info(f"Subscribed to '{topic}'")

        # Timer to keep GUI alive
        self.create_timer(0.1, self._spin_plots)

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # Draw reference frame if loaded
        if self.T_ref:
            self._draw_frame(self.T_ref, 'ref', length=0.5)
        self.dynamic_artist = []
        plt.ion()
        plt.show()

    def _draw_frame(self, T: Transform, label: str, length=0.3):
        # Build homogeneous matrix
        q   = [T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w]
        mat = quaternion_matrix(q)
        mat[0:3, 3] = [T.translation.x, T.translation.y, T.translation.z]

        origin = mat[0:3, 3]
        axes   = mat[0:3, :3] * length
        colors = ['r', 'g', 'b']
        for i, c in enumerate(colors):
            self.ax.quiver(
                *origin,
                *axes[:, i],
                color=c,
                linewidth=2,
                arrow_length_ratio=0.1
            )
        self.ax.text(*(origin + 0.05), label, size=10, zorder=1)

    def tf_callback(self, msg: TFMessage):
        # If there are any transforms, take the first one
        if not msg.transforms:
            return

        first = msg.transforms[0]
        self.latest_dyn = first.transform

        # Clear and redraw plot
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # Draw reference if present
        if self.T_ref:
            self._draw_frame(self.T_ref, 'ref', length=0.5)

        # Draw dynamic frame
        T = first.transform
        q   = [T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w]
        mat = quaternion_matrix(q)
        mat[0:3, 3] = [T.translation.x, T.translation.y, T.translation.z]

        origin = mat[0:3, 3]
        axes   = mat[0:3, :3] * 0.3
        colors = ['r', 'g', 'b']
        for i, c in enumerate(colors):
            art = self.ax.quiver(
                *origin,
                *axes[:, i],
                color=c,
                linewidth=2,
                arrow_length_ratio=0.1
            )
            self.dynamic_artist.append(art)

        text = self.ax.text(
            *(origin + 0.05),
            first.child_frame_id,
            size=10,
            zorder=1
        )
        self.dynamic_artist.append(text)

        self.fig.canvas.draw()

    def _on_key(self, event):
        # On 'S', save latest_dyn back to YAML
        if event.key == 'S' and self.latest_dyn is not None:
            out = {
                'transform': {
                    'translation': {
                        'x': self.latest_dyn.translation.x,
                        'y': self.latest_dyn.translation.y,
                        'z': self.latest_dyn.translation.z,
                    },
                    'rotation': {
                        'x': self.latest_dyn.rotation.x,
                        'y': self.latest_dyn.rotation.y,
                        'z': self.latest_dyn.rotation.z,
                        'w': self.latest_dyn.rotation.w,
                    }
                }
            }
            try:
                with open(self.ref_yaml, 'w') as f:
                    yaml.dump(out, f)
                # Load into T_ref
                self.T_ref = self.latest_dyn
                self.get_logger().info(
                    f"Saved and loaded new reference transform to '{self.ref_yaml}'"
                )
            except Exception as e:
                self.get_logger().error(
                    f"Failed to save reference YAML: {e}"
                )
        # On 'Q', quit
        elif event.key == 'Q':
            self.get_logger().info("Quit key pressed, shutting down...")
            raise KeyboardInterrupt

    def _spin_plots(self):
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = FramePlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close(node.fig)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()