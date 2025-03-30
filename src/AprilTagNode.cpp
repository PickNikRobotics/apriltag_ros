// ros
#include "pose_estimation.hpp"
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include "image_transport/subscriber_filter.hpp"
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <realsense2_camera_msgs/realsense2_camera_msgs/msg/rgbd.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions& options);

    ~AprilTagNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<bool> profile;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    std::function<void(apriltag_family_t*)> tf_destructor;

    // Subscribers
    // Depth not enabled (default)
    image_transport::CameraSubscriber sub_cam;
    // Depth enabled
    std::shared_ptr<rclcpp::Subscription<realsense2_camera_msgs::msg::RGBD>> sub_cam_rgbd;
    // Objects for transforming between depth and rgb frames
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
    std::shared_ptr<geometry_msgs::msg::TransformStamped> depth_to_rgb_tf;

    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    tf2_ros::TransformBroadcaster tf_broadcaster;

    pose_estimation_f estimate_pose = nullptr;

    void onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
                  const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);
    void onCameraRGBD(const realsense2_camera_msgs::msg::RGBD::SharedPtr rgbd_msg);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    tf_buffer(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
    tf_listener(std::make_shared<tf2_ros::TransformListener>(*tf_buffer)),
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    tf_broadcaster(this)
{
    const std::string transport = declare_parameter("image_transport", "raw", descr({}, true));

    if(declare_parameter("use_rgbd", false, descr("enabled use of aligned depth topic", true))) {
        // Determine if we are using depth information or not.
        sub_cam_rgbd = this->create_subscription<realsense2_camera_msgs::msg::RGBD>(
            "rgbd", rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_sensor_data)),
            std::bind(&AprilTagNode::onCameraRGBD, this, std::placeholders::_1));
    }
    else {
        // Use image_transport tool to create synchronized subscription between camera_info & rgb.
        sub_cam = image_transport::create_camera_subscription(
            this, this->get_node_topics_interface()->resolve_topic_name("image_rect"),
            std::bind(&AprilTagNode::onCamera, this, std::placeholders::_1, std::placeholders::_2), transport,
            rmw_qos_profile_sensor_data);
    }

    // read-only parameters
    const std::string tag_family = declare_parameter("family", "36h11", descr("tag family", true));
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size", true));

    // get tag names, IDs and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

    // get method for estimating tag pose
    estimate_pose = pose_estimation_methods.at(declare_parameter("pose_estimation_method", "pnp", descr("pose estimation method: \"pnp\" (more accurate) or \"homography\" (faster)", true)));

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
    }

    if(tag_fun.count(tag_family)) {
        tf = tag_fun.at(tag_family).first();
        tf_destructor = tag_fun.at(tag_family).second;
        apriltag_detector_add_family(td, tf);
    }
    else {
        throw std::runtime_error("Unsupported tag family: " + tag_family);
    }
}

AprilTagNode::~AprilTagNode()
{
    apriltag_detector_destroy(td);
    tf_destructor(tf);
}

void AprilTagNode::onCameraRGBD(const realsense2_camera_msgs::msg::RGBD::SharedPtr rgbd_msg)
{
    RCLCPP_INFO_ONCE(get_logger(), "Using realsense RGBD topic for apriltag detections.");
    if(depth_to_rgb_tf) {
        // Already loaded the transform.
    }
    else if(tf_buffer->canTransform(rgbd_msg->rgb.header.frame_id, rgbd_msg->depth.header.frame_id,
                                    tf2::TimePointZero)) {
        depth_to_rgb_tf = std::make_shared<geometry_msgs::msg::TransformStamped>(tf_buffer->lookupTransform(
            rgbd_msg->rgb.header.frame_id, rgbd_msg->depth.header.frame_id, tf2::TimePointZero));
    }
    else {
        RCLCPP_WARN(get_logger(),
                    "No transform available between depth and color frames, returning without detecting.");
        return;
    }

    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {
        rgbd_msg->rgb_camera_info.p.data()[0], rgbd_msg->rgb_camera_info.p.data()[5],
        rgbd_msg->rgb_camera_info.p.data()[2], rgbd_msg->rgb_camera_info.p.data()[6]};

    // convert to 8bit monochrome image
    const cv::Mat img_uint8 = cv_bridge::toCvCopy(rgbd_msg->rgb, "mono8")->image;

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = rgbd_msg->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        double cx = det->c[0];
        double cy = det->c[1];
        // Get depth point (z) and transform to rgb frame where the other points are represented.
        cv::Mat depth_image = cv_bridge::toCvCopy(rgbd_msg->depth, rgbd_msg->depth.encoding)->image;
        float depth = depth_image.at<uint16_t>(static_cast<int>(std::round(cy)), static_cast<int>(std::round(cx))) *
                      0.001f;  // mm to meters
        geometry_msgs::msg::PointStamped depth_point, depth_point_rgb_frame;
        depth_point.header.frame_id = rgbd_msg->depth.header.frame_id;
        depth_point.point.z = depth;
        tf2::doTransform(depth_point, depth_point_rgb_frame, *depth_to_rgb_tf);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        geometry_msgs::msg::TransformStamped tf;
        tf.header = rgbd_msg->rgb.header;
        // set child frame name by generic tag name or configured tag name
        tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
        const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
        if(estimate_pose != nullptr) {
            tf.transform = estimate_pose(det, intrinsics, size);
        }

        if (depth > 0.0 && !std::isnan(depth)) {
            tf.transform.translation.z = depth_point_rgb_frame.point.z;
        } else {
            RCLCPP_WARN(this->get_logger(), "Invalid depth at center, using rgb detection.");
        }
        tfs.push_back(tf);
    }

    pub_detections->publish(msg_detections);
    tf_broadcaster.sendTransform(tfs);

    apriltag_detections_destroy(detections);
}

void AprilTagNode::onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
                            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {msg_ci->p.data()[0], msg_ci->p.data()[5], msg_ci->p.data()[2], msg_ci->p.data()[6]};

    // convert to 8bit monochrome image
    const cv::Mat img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        geometry_msgs::msg::TransformStamped tf;
        tf.header = msg_img->header;
        // set child frame name by generic tag name or configured tag name
        tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
        const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
        if(estimate_pose != nullptr) {
            tf.transform = estimate_pose(det, intrinsics, size);
        }

        tfs.push_back(tf);
    }

    pub_detections->publish(msg_detections);
    tf_broadcaster.sendTransform(tfs);

    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}
