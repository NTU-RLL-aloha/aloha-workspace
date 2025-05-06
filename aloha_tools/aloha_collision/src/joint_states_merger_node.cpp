// src/joint_state_merger.cpp

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>

using sensor_msgs::msg::JointState;

class JointStateMerger : public rclcpp::Node
{
public:
  JointStateMerger()
      : Node("joint_state_merger")
  {
    // Declare parameters
    this->declare_parameter<std::vector<std::string>>("input_prefixes", {});
    this->declare_parameter<std::string>("output_prefix", "");

    // Get parameters
    this->get_parameter("input_prefixes", input_prefixes_);
    this->get_parameter("output_prefix", output_prefix_);

    if (input_prefixes_.empty())
    {
      RCLCPP_ERROR(get_logger(),
                   "Parameter 'input_prefixes' is empty; at least one prefix is required.");
      throw std::runtime_error("No input prefixes provided");
    }
    if (output_prefix_.empty())
    {
      RCLCPP_ERROR(get_logger(),
                   "Parameter 'output_prefix' is empty; please specify a valid output prefix.");
      throw std::runtime_error("No output prefix provided");
    }

    // Construct output topic and create publisher
    std::string output_topic = "/" + output_prefix_ + "/joint_states";
    pub_ = create_publisher<JointState>(output_topic, 10);
    RCLCPP_INFO(get_logger(),
                "Publishing merged JointState on '%s'",
                output_topic.c_str());

    // Subscribe to each input prefix's joint_states topic
    for (const auto &prefix : input_prefixes_)
    {
      std::string topic = "/" + prefix + "/joint_states";
      auto sub = create_subscription<JointState>(
          topic, 10,
          [this, prefix](const JointState::SharedPtr msg)
          {
            this->callback(msg, prefix);
          });
      subs_.push_back(sub);
      RCLCPP_INFO(get_logger(),
                  "Subscribed to '%s' for prefix '%s'",
                  topic.c_str(), prefix.c_str());
    }
  }

private:
  void callback(const JointState::SharedPtr msg, const std::string &prefix)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Build a local copy with prefixed joint names
    JointState local;
    local.header = msg->header;
    for (size_t i = 0; i < msg->name.size(); ++i)
    {
      std::string name = prefix + "/" + msg->name[i];
      double pos = i < msg->position.size() ? msg->position[i] : 0.0;
      double vel = i < msg->velocity.size() ? msg->velocity[i] : 0.0;
      double eff = i < msg->effort.size() ? msg->effort[i] : 0.0;

      local.name.push_back(name);
      local.position.push_back(pos);
      local.velocity.push_back(vel);
      local.effort.push_back(eff);
    }

    // Merge into the master message
    if (merged_.name.empty())
    {
      // First message initializes merged_
      merged_ = local;
      build_index_map();
    }
    else
    {
      // Update or append each joint
      for (size_t i = 0; i < local.name.size(); ++i)
      {
        const auto &name = local.name[i];
        auto it = name_to_index_.find(name);
        if (it == name_to_index_.end())
        {
          // New joint → append
          name_to_index_[name] = merged_.name.size();
          merged_.name.push_back(name);
          merged_.position.push_back(local.position[i]);
          merged_.velocity.push_back(local.velocity[i]);
          merged_.effort.push_back(local.effort[i]);
        }
        else
        {
          // Existing joint → update
          size_t idx = it->second;
          merged_.position[idx] = local.position[i];
          if (idx < merged_.velocity.size())
            merged_.velocity[idx] = local.velocity[i];
          if (idx < merged_.effort.size())
            merged_.effort[idx] = local.effort[i];
        }
      }
    }

    // Stamp and publish
    merged_.header.stamp = now();
    pub_->publish(merged_);
  }

  void build_index_map()
  {
    name_to_index_.clear();
    for (size_t i = 0; i < merged_.name.size(); ++i)
    {
      name_to_index_[merged_.name[i]] = i;
    }
  }

  // Parameters
  std::vector<std::string> input_prefixes_;
  std::string output_prefix_;

  // ROS interfaces
  rclcpp::Publisher<JointState>::SharedPtr pub_;
  std::vector<rclcpp::Subscription<JointState>::SharedPtr> subs_;

  // Merging state
  JointState merged_;
  std::unordered_map<std::string, size_t> name_to_index_;
  std::mutex mutex_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  try
  {
    auto node = std::make_shared<JointStateMerger>();
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("rclcpp"),
                 "Failed to start joint_state_merger: %s", e.what());
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}