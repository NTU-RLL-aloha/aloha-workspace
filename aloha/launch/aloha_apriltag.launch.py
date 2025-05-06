from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    pattern_launch_arg = LaunchConfiguration('pattern')
    tag_launch_arg = LaunchConfiguration('tag')
    
    pattern_config_path_launch_arg = PathJoinSubstitution([
        FindPackageShare('aloha'),
        'config',
        'apriltag',
        'patterns',
        pattern_launch_arg.perform(context) + '.yaml',
    ])
    tag_config_path_launch_arg = PathJoinSubstitution([
        FindPackageShare('aloha'),
        'config',
        'apriltag',
        'tags',
        tag_launch_arg.perform(context) + '.yaml',
    ])
    
    apriltag_node = Node(
        package="apriltag_ros",
        executable="apriltag_node",
        name="apriltag_node",
        output="log",
        parameters=[
            tag_config_path_launch_arg
        ],
        remappings=[
            ('/image_rect', LaunchConfiguration('image_rect')),
            ('/camera_info', LaunchConfiguration('camera_info')),
        ],
        # arguments={
        #     'image_rect': LaunchConfiguration('image_rect'),
        #     'camera_info': LaunchConfiguration('camera_info'),
        # }.items(),
    )

    apriltag_localization_node = Node(
        package='aloha_camera',
        executable='apriltag_localization_node',
        name='apriltag_localization_node',
        output='log',
        parameters=[
            tag_config_path_launch_arg,
            pattern_config_path_launch_arg,
        ],
    )

    return [
        apriltag_node,
        apriltag_localization_node,
    ]
    
    
def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'pattern',
            default_value='mid_1x1',
            description='The pattern of the AprilTag arrangement',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'tag',
            default_value='36h11',
            description='The type of AprilTag to use',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'image_rect',
            default_value='/camera/image',
            description='Topic name of the rectified image topic',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'camera_info',
            default_value='/camera/camera_info',
            description='Topic name of the camera info topic',
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
