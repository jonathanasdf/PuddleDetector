#include <cassert>
#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

using namespace boost::filesystem;
using namespace cv;
using namespace pcl;
using namespace std;
typedef double ts;
typedef PointCloud<PointXYZ> Cloud;
typedef shared_ptr<Eigen::Matrix4d> pose_p;
typedef shared_ptr<const Eigen::Matrix4d> const_pose_p;

/************************* GLOBAL VARIABLES ************************/
// Configuration
const int num_frames_to_keep = 10;

// thresholds
const double lidar_near_sqr_thresh = 1,
             ground_distance_thresh = 0.1;

// Data paths
map<ts, path> left_img_paths, right_img_paths, lidar_paths;
const string pose_path = "data/odom_clean.dat";

// extrinsic calibration
Eigen::Matrix4d T_vehicle_lidar       {{0,      -1,       0,  0.0170},
                                       {1,       0,       0, -0.0270},
                                       {0,       0,       1,  0.0370},
                                       {0,       0,       0,       1}};

Eigen::Matrix4d T_vehicle_camera_left {{1,       0,       0, -0.0140},
                                       {0, -0.2215,  0.9751,  0.0282},
                                       {0, -0.9751, -0.2215, -0.0100},
                                       {0,       0,       0,       1}};

// intrinsic calibration
const double focal_length = 469.1630, // pixels
                       cx = 508.5,
                       cy = 285.5;
Eigen::Matrix<double, 3, 4> T_projection {{focal_length, 0, cx, 0},
                                          {0, focal_length, cy, 0},
                                          {0, 0,             1, 0}};

// poses
map<ts, pose_p> poses;

/*********************** END GLOBAL VARIABLES **********************/

// string to timestamp
ts stots(string s) {
    return stod(s);
}
void loadData() {
    for (auto i = directory_iterator(path("data")), end = directory_iterator(); i != end; i++) {
        path file = i->path();
        string filename = file.filename().string();
        if (boost::starts_with(filename, "left")) {
            left_img_paths[stots(filename.substr(11, 20))] = file;
        } else if (boost::starts_with(filename, "right")) {
            right_img_paths[stots(filename.substr(12, 20))] = file;
        } else if (boost::starts_with(filename, "lidar")) {
            lidar_paths[stots(filename.substr(12, 20))] = file;
        }
    }

    std::ifstream pose_in(pose_path);
    ts t;
    while(pose_in >> t) {
        auto T = make_shared<Eigen::Matrix4d>();
        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                pose_in >> (*T)(i,j);
            }
        }
        poses[t] = move(T);
    }
    pose_in.close();
    cout << "Data load success!" << endl;
}
template<typename T> T getClosestFrame(ts frame, map<ts, T> &map) {
    auto ptr = map.upper_bound(frame);
    if (ptr == map.begin()) return ptr->second;
    auto ptr2 = ptr--;
    ts d1 = abs(frame - ptr->first);
    ts d2 = abs(frame - ptr2->first);
    return d1 < d2 ? ptr->second : ptr2->second;
}
// Get transform from lidar frame to global frame
Eigen::Matrix4d T_lidar_global(const_pose_p pose) {
    return *pose * T_vehicle_lidar.inverse();
}
// Get transform from global frame to image pixel
Eigen::Matrix<double, 3, 4> T_global_image(const_pose_p pose) {
    return T_projection * T_vehicle_camera_left * pose->inverse();
}
void getGroundPlane(Cloud::ConstPtr in_cloud,
                    Cloud::Ptr ground_cloud,
                    Cloud::Ptr stuff_cloud) {
   ModelCoefficients coefficients;
   PointIndices::Ptr inliers(new PointIndices);
   // segment it!
   SACSegmentation<PointXYZ> seg;
   seg.setOptimizeCoefficients(true);
   seg.setModelType(SACMODEL_PLANE);
   seg.setMethodType(SAC_RANSAC);
   seg.setDistanceThreshold(ground_distance_thresh);
   seg.setInputCloud(in_cloud);
   seg.segment(*inliers, coefficients);

   // extract the plane into a new point cloud
   ExtractIndices<PointXYZ> extract;
   extract.setInputCloud(in_cloud);
   extract.setIndices(inliers);
   extract.setNegative(false);
   extract.filter(*ground_cloud);
   extract.setNegative(true);
   extract.filter(*stuff_cloud);
}
Mat processFrame(const deque<Mat> &camera_frames,
                 const deque<const_pose_p> &pose_frames,
                 const deque<Cloud::ConstPtr> &lidar_frames) {
    Mat out(camera_frames[0]);
    Cloud::Ptr ground(new Cloud), stuff(new Cloud);
    getGroundPlane(lidar_frames[0], ground, stuff);
    vector<Point> pixels, hull;
    for (auto pt : ground->points) {
        auto pose = pose_frames[0];
        Eigen::Vector3d pixel_hom = T_global_image(pose) * T_lidar_global(pose) * pt.getVector4fMap().cast<double>();
        Point2d pixel(pixel_hom(0)/pixel_hom(2), pixel_hom(1)/pixel_hom(2));
        pixels.push_back(pixel);
    }
    convexHull(pixels, hull);
    for (int j=0; j < out.rows; j++) {
        for (int i=0; i < out.cols; i++) {
            if (pointPolygonTest(hull, Point2d(i, j), false) >= 0) {
                auto &color = out.at<Vec3b>(j, i);
                color[0] = 255;
                color[1] = 0;
                color[2] = 0;
            }
        }
    }
    return out;
}

int main(int argc, char **argv) {
    loadData();

    char video[] = "video";
    cvNamedWindow(video);

    deque<Mat> camera_frames;
    deque<const_pose_p> pose_frames;
    deque<Cloud::ConstPtr> lidar_frames;

    ts last_frame_timestamp = -1;
    auto last_frame_time = chrono::high_resolution_clock::now();
    for(auto p : left_img_paths) {
        // frame timing
        ts frame = p.first;
        auto current_time = chrono::high_resolution_clock::now();
        if (last_frame_timestamp != -1) {
            auto timestamp_diff = frame - last_frame_timestamp;
            auto time_diff = chrono::duration<double>(current_time - last_frame_time).count();
            auto wait = max(1., (timestamp_diff - time_diff) * 1000);
            cvWaitKey(wait);
        }
        last_frame_timestamp = frame;
        last_frame_time = chrono::high_resolution_clock::now();
        cout << fixed << "Current frame: " << frame << endl;

        // read camera image
        auto camera = imread(left_img_paths[frame].string());
        camera_frames.emplace_front(camera);
        if (camera_frames.size() > num_frames_to_keep) camera_frames.pop_back();

        // read pose
        auto pose = getClosestFrame(frame, poses);
        pose_frames.push_front(pose);
        if (pose_frames.size() > num_frames_to_keep) pose_frames.pop_back();
        cout << "Current pose: " << endl << *pose << endl;

        // read lidar
        auto lidar_path = getClosestFrame(frame, lidar_paths);
        Cloud::Ptr lidar(new Cloud);
        io::loadPCDFile<PointXYZ>(lidar_path.string(), *lidar);

        assert(lidar.is_dense);
        Cloud::Ptr lidar_filtered(new Cloud);
        for (auto pt : lidar->points) {
            if (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z < lidar_near_sqr_thresh) continue;
            lidar_filtered->push_back(pt);
        }

        lidar_frames.push_front(lidar_filtered);
        if (lidar_frames.size() > num_frames_to_keep) lidar_frames.pop_back();
        cout << lidar->points.size() - lidar_filtered->points.size() << " points filtered." << endl;

        // process frame
        auto img = processFrame(camera_frames, pose_frames, lidar_frames);
        imshow(video, img);
    }
    cvWaitKey();
    return 0;
}
