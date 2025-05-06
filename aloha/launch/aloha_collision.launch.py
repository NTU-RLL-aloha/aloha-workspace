import yaml

from aloha.moveit_description import (
    declare_interbotix_xsarm_robot_description_launch_arguments,
    declare_interbotix_xsarm_robot_description_semantic_launch_arguments,
)
from aloha.utils import prefix_xml, merge_urdf, merge_srdf, dumps_et
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
    moveit_description_follower_left_launch_arg = LaunchConfiguration(
        'moveit_description_follower_left'
    )
    moveit_description_follower_right_launch_arg = LaunchConfiguration(
        'moveit_description_follower_right'
    )
    
    moveit_description_semantic_follower_left_launch_arg = LaunchConfiguration(
        'moveit_description_semantic_follower_left'
    )
    moveit_description_semantic_follower_right_launch_arg = LaunchConfiguration(
        'moveit_description_semantic_follower_right'
    )

    description_root_path = PathJoinSubstitution([
        FindPackageShare('aloha'),
        'config',
        'moveit',
        'urdf',
        'aloha_root.urdf',
    ])
    prefixed_robot_descriptions = prefix_xml(
        inputs=[
            ('follower_left', moveit_description_follower_left_launch_arg.perform(context)),
            ('follower_right', moveit_description_follower_right_launch_arg.perform(context)),
        ],
        targets=[
            (["joint"], ["name"]),
            (["mimic"], ["joint"]),
        ],
    )
    # TODO: Modify the transforms to be relative to the base link frame
    transform_config_path = PathJoinSubstitution([
        FindPackageShare('aloha'),
        'config',
        'moveit',
        'aloha_transform.yaml',
    ]).perform(context)
    transform_config = yaml.load(open(transform_config_path, 'r'))

    merged_urdf = merge_urdf(
        template_path=description_root_path.perform(context),
        children=prefixed_robot_descriptions,
        transforms=transform_config,
        base_link_frame=LaunchConfiguration('base_link_frame').perform(context),
    )
    merged_robot_description = dumps_et(merged_urdf)

    description_semantic_root_path = PathJoinSubstitution([
        FindPackageShare('aloha'),
        'config',
        'moveit',
        'srdf',
        'aloha_root.srdf',
    ])
    prefixed_robot_description_semantic = prefix_xml(
        inputs=[
            ('follower_left', moveit_description_semantic_follower_left_launch_arg.perform(context)),
            ('follower_right', moveit_description_semantic_follower_right_launch_arg.perform(context)),
        ],
        targets = [
            (["joint"], ["name"]),
            (["group"], ["name"]),
            (["group_state"], ["group"]),
            (["end_effector"], ["group"]),
        ],
    )
    merged_srdf = merge_srdf(
        template_path=description_semantic_root_path.perform(context),
        children=prefixed_robot_description_semantic,
    )
    merged_robot_description_semantic = dumps_et(merged_srdf)
    
    joint_states_topic_prefix_launch_arg = LaunchConfiguration(
        'joint_states_topic_prefix',
    )
    collision_detection_node = Node(
        package='aloha_collision',
        executable='collision_detection_node',
        name='collision_detection',
        output='log',
        parameters=[{
            'robot_description': merged_robot_description,
            'robot_description_semantic': merged_robot_description_semantic,
            'joint_states_topic_prefix': joint_states_topic_prefix_launch_arg,
        }],
    )
    
    robot_name_follower_left = LaunchConfiguration(
        'robot_name_follower_left',
    ).perform(context)
    robot_name_follower_right = LaunchConfiguration(
        'robot_name_follower_right',
    ).perform(context)
    joint_states_merger_node = Node(
        package='aloha_collision',
        executable='joint_states_merger_node',
        name='joint_states_merger',
        output='log',
        parameters=[{
            'input_prefixes': [
                robot_name_follower_left, 
                robot_name_follower_right
            ],
            'output_prefix': LaunchConfiguration('joint_states_topic_prefix'),
        }],
    )

    return [
        collision_detection_node,
        joint_states_merger_node,
    ]
    
    
def generate_launch_description():
    declared_arguments = []
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='moveit_description_follower_left',
            robot_model_launch_config_name='robot_model_follower',
            robot_name_launch_config_name='robot_name_follower_left',
            base_link_frame='base_link',
            use_world_frame='false',
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_launch_arguments(
            robot_description_launch_config_name='moveit_description_follower_right',
            robot_model_launch_config_name='robot_model_follower',
            robot_name_launch_config_name='robot_name_follower_right',
            base_link_frame='base_link',
            use_world_frame='false',
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_semantic_launch_arguments(
            robot_description_semantic_launch_config_name='moveit_description_semantic_follower_left',
            robot_model_launch_config_name='robot_model_follower',
            robot_name_launch_config_name='robot_name_follower_left',
            base_link_frame='base_link',
        )
    )
    declared_arguments.extend(
        declare_interbotix_xsarm_robot_description_semantic_launch_arguments(
            robot_description_semantic_launch_config_name='moveit_description_semantic_follower_right',
            robot_model_launch_config_name='robot_model_follower',
            robot_name_launch_config_name='robot_name_follower_right',
            base_link_frame='base_link',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            'joint_states_topic_prefix',
            default_value='collision',
            description='The prefix of topic to subscribe to for joint states',
        )
    )

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
