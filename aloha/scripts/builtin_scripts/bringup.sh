source /opt/ros/humble/setup.bash # configure ROS system install environment
source ~/interbotix_ws/install/setup.bash # configure ROS workspace environment
ros2 launch aloha aloha_bringup.launch.py robot:=aloha_mobile # launch hardware drivers and control software