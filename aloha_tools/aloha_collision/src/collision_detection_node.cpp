#include <sstream>
#include <limits>

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <moveit/moveit_cpp/moveit_cpp.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/collision_detection/collision_common.h>

// #define WARN_COLLISION

void check_collision(collision_detection::CollisionRequest &collision_request,
                     collision_detection::CollisionResult &collision_result,
                     const rclcpp::Node::SharedPtr node,
                     const planning_scene::PlanningScenePtr &scene,
                     const collision_detection::AllowedCollisionMatrix &acm)
{
  auto collision_env = scene->getCollisionEnv();
  collision_env->checkCollision(collision_request, collision_result, scene->getCurrentState(), acm);

  if (collision_result.collision)
  {
    // Collect colliding link names
    std::vector<std::string> colliding_links;
    for (const auto &contact_map_pair : collision_result.contacts)
    {
      const auto &contact_pair = contact_map_pair.first;
      const auto &contact = contact_map_pair.second;
      colliding_links.push_back(contact_pair.first);
      colliding_links.push_back(contact_pair.second);
    }
    // Log unique link names
    std::sort(colliding_links.begin(), colliding_links.end());
    colliding_links.erase(std::unique(colliding_links.begin(), colliding_links.end()), colliding_links.end());
// #ifdef WARN_COLLISION
    // TODO: Send collision topic
    RCLCPP_WARN(node->get_logger(),
                "Collision detected between links: %s",
                std::accumulate(std::next(colliding_links.begin()), colliding_links.end(),
                                colliding_links.front(),
                                [](const std::string &a, const std::string &b)
                                { return a + ", " + b; })
                    .c_str());
// #endif
    collision_result.clear();
  }
}

void check_distance(collision_detection::DistanceRequest &distance_request,
                    collision_detection::DistanceResult &distance_result,
                    const rclcpp::Node::SharedPtr node,
                    const planning_scene::PlanningScenePtr &scene,
                    const collision_detection::AllowedCollisionMatrix &acm,
                    const double threshold)
{
  auto collision_env = scene->getCollisionEnv();
  auto robot_model = collision_env->getRobotModel();

  distance_request.enableGroup(robot_model);
  distance_request.acm = &acm;

  collision_env->distanceSelf(distance_request, distance_result, scene->getCurrentState());

  for (const auto &distance_pair : distance_result.distances)
  {
    const auto &pair = distance_pair.first;
    const auto &link1 = pair.first, &link2 = pair.second;
    const auto &distances = distance_pair.second;
    auto min_it = std::min_element(distances.begin(), distances.end(),
                                   [](const auto &a, const auto &b)
                                   {
                                     return a.distance < b.distance;
                                   });
    double distance = min_it != distances.end() ? min_it->distance : std::numeric_limits<double>::max();
    if (distance < threshold)
    {
      // TODO: Send collision topic
      RCLCPP_WARN(node->get_logger(),
                  "%s and %s is too close with distance %f",
                  link1.c_str(), link2.c_str(), distance);
    }
  }
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("collision_detection_node");

  // Declare and retrieve the 'robot_description' and 'robot_description_semantice' parameter (URDF/SRDF XML)
  node->declare_parameter<std::string>("robot_description", "");
  node->declare_parameter<std::string>("robot_description_semantic", "");
  // Declare the topic name for joint states
  node->declare_parameter<std::string>("joint_states_topic_prefix", "");
  // Distance configuration
  node->declare_parameter<bool>("check_distance", false);
  node->declare_parameter<double>("distance_ignorance", 0.1);
  node->declare_parameter<double>("distance_threshold", 0.05);

  std::string robot_description_;
  std::string robot_description_semantic_;
  node->get_parameter("robot_description", robot_description_);
  node->get_parameter("robot_description_semantic", robot_description_semantic_);

  if (robot_description_.empty())
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Parameter 'robot_description' is empty. "
                 "Please pass your URDF content.");
    return 1;
  }
  if (robot_description_semantic_.empty())
  {
    RCLCPP_ERROR(node->get_logger(),
                 "Parameter 'robot_description_semantic' is empty. "
                 "Please pass your SRDF content.");
    return 1;
  }

  // Verify we can load the model
  auto rml_opts = robot_model_loader::RobotModelLoader::Options(robot_description_, robot_description_semantic_);
  auto robot_model_loader = std::make_shared<robot_model_loader::RobotModelLoader>(node, rml_opts);
  auto psm = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node, robot_model_loader);

  std::string joint_states_topic_prefix_;
  node->get_parameter("joint_states_topic_prefix", joint_states_topic_prefix_);
  std::string joint_states_topic = "/" + joint_states_topic_prefix_ + "/joint_states";

  psm->startStateMonitor(joint_states_topic);
  // psm->startWorldGeometryMonitor();
  psm->startSceneMonitor();

  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // Log the allowed collision matrix
  auto scene = psm->getPlanningScene();
  const auto &acm = scene->getAllowedCollisionMatrix();

  std::vector<std::string> names;
  acm.getAllEntryNames(names); // all link names in the model
  RCLCPP_DEBUG(node->get_logger(),
               "Allowed collision matrix size: %zu", acm.getSize());
  for (const std::string &name : names)
    RCLCPP_DEBUG(node->get_logger(),
                 "Name: %s", name.c_str());

  collision_detection::AllowedCollision::Type type;
  for (const std::string &a : names)
    for (const std::string &b : names)
      if (acm.getEntry(a, b, type))
        if (type == collision_detection::AllowedCollision::ALWAYS)
          RCLCPP_DEBUG(node->get_logger(),
                       "Always allowed collision: %s <-> %s", a.c_str(), b.c_str());
        else if (type == collision_detection::AllowedCollision::NEVER)
          RCLCPP_DEBUG(node->get_logger(),
                       "Never allowed collision: %s <-> %s", a.c_str(), b.c_str());
        else if (type == collision_detection::AllowedCollision::CONDITIONAL)
          RCLCPP_DEBUG(node->get_logger(),
                       "Conditional allowed collision: %s <-> %s", a.c_str(), b.c_str());

  RCLCPP_INFO(node->get_logger(), "Global PSM node initialized, entering collision-check loop.");

  // Set to default states
  // auto robot_model = robot_model_loader->getModel();
  // moveit::core::RobotState state(robot_model);
  // {
  //   auto *jmg = state.getJointModelGroup("follower_left/arm");
  //   state.setToDefaultValues(jmg, "Home");
  // }
  // {
  //   auto *jmg = state.getJointModelGroup("follower_right/arm");
  //   state.setToDefaultValues(jmg, "Home");
  // }

  // configure req for self‐collision and all links
  collision_detection::CollisionRequest collision_request;
  collision_detection::CollisionResult collision_result;

  collision_request.distance = true;
  collision_request.contacts = true;
  collision_request.group_name = ""; // check all groups

  // configure req for distance and all links
  bool check_distance_;
  double distance_ignorance_;
  double distance_threshold_;
  node->get_parameter("check_distance", check_distance_);
  node->get_parameter("distance_ignorance", distance_ignorance_);
  node->get_parameter("distance_threshold", distance_threshold_);

  collision_detection::DistanceRequest distance_request;
  collision_detection::DistanceResult distance_result;
  distance_request.type = collision_detection::DistanceRequestType::ALL;
  distance_request.enable_signed_distance = true;
  distance_request.distance_threshold = distance_threshold_;

  // Main loop: check collisions at 10 Hz
  rclcpp::Rate rate(10.0);
  while (rclcpp::ok())
  {
    auto scene = psm->getPlanningScene();
    if (check_distance_)
      check_distance(distance_request, distance_result, node, scene, acm, distance_ignorance_);
    else
      check_collision(collision_request, collision_result, node, scene, acm);
    
    rclcpp::spin_some(node);
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}