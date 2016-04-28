#include <cassert>
#include <fstream>
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

using namespace boost::filesystem;
using namespace cv;
using namespace pcl;
using namespace std;
typedef double ts;
typedef shared_ptr<Eigen::Matrix4d> pose_p;

/************************* GLOBAL VARIABLES ************************/
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

// thresholds
const double lidar_near_sqr_thresh = 1,
             ground_distance_thresh = 0.1;
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
Eigen::Matrix4d T_lidar_global(pose_p pose) {
  return *pose * T_vehicle_lidar.inverse();
}
// Get transform from global frame to image pixel
Eigen::Matrix<double, 3, 4> T_global_image(pose_p pose) {
  return T_projection * T_vehicle_camera_left * pose->inverse();
}

void getGroundPlane(PointCloud<PointXYZ>::Ptr in_cloud, PointCloud<PointXYZ>::Ptr out_cloud) {
    ModelCoefficients coefficients;
    PointIndices inliers;
    // segment it!
    SACSegmentation<PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SACMODEL_PLANE);
    seg.setMethodType(SAC_RANSAC);
    seg.setDistanceThreshold(ground_distance_thresh);
    seg.setInputCloud(in_cloud);
    seg.segment(inliers, coefficients);

    // extract the plane into a new point cloud
    ExtractIndices extract;
    extract.setInputCloud(in_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*out_cloud);
}

int main(int argc, char **argv) {
    loadData();

    char video[] = "video";
    cvNamedWindow(video);

    ts last_frame_time = -1;
    for(auto p : left_img_paths) {
        ts frame = p.first;
        if (!right_img_paths.count(frame)) continue;

        if (last_frame_time != -1) {
          cvWaitKey((frame - last_frame_time) * 1000);
        }
        last_frame_time = frame;
        cout << "Current time: " << frame << endl;

        auto left = imread(left_img_paths[frame].string());
        auto right = imread(left_img_paths[frame].string());
        assert(left.rows == right.rows);
        assert(left.type() == right.type());
        Mat combined(left.rows, left.cols + right.cols, left.type());
        left.copyTo(combined(Rect(0, 0, left.cols, left.rows)));
        right.copyTo(combined(Rect(left.cols, 0, right.cols, right.rows)));
        imshow(video, combined);

        auto lidar_path = getClosestFrame(frame, lidar_paths);
        PointCloud<PointXYZ>::Ptr lidar (new PointCloud<PointXYZ>);
        io::loadPCDFile<PointXYZ> (lidar_path.string(), *lidar);
        cout << lidar->points.size() << " points loaded." << endl;

        auto pose = getClosestFrame(frame, poses);
        cout << "Current pose: " << *pose << endl;

        assert(lidar.is_dense);
        for (auto pt : lidar->points) {
          if (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z > lidar_near_sqr_thresh) continue;
          Eigen::Vector3d pixel = T_global_image(pose) * T_lidar_global(pose) * pt.getVector4fMap().cast<double>();
        }
    }
    cvWaitKey();
    return 0;
}
