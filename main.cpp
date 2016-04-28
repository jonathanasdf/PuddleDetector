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

#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace boost::filesystem;
using namespace cv;
using namespace pcl;
using namespace std;
typedef double ts;
typedef shared_ptr<Eigen::Matrix4d> pose_p;
typedef shared_ptr<const Eigen::Matrix4d> const_pose_p;
typedef PointCloud<PointXYZ> Cloud;

/************************* GLOBAL VARIABLES ************************/
// Configuration
const int num_frames_to_keep = 1;
const double voxel_size = 0.1; // m or lidar units

// thresholds
const double lidar_near_sqr_thresh = 4, // m^2
             ground_distance_thresh = 0.3; // m

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

// intrinsic calibration and sensor properties
const double focal_length = 469.1630, // pixels
                       cx = 508.5, // pixels
                       cy = 285.5; // pixels

Eigen::Matrix<double, 3, 4> T_projection {{focal_length, 0, cx, 0},
                                          {0, focal_length, cy, 0},
                                          {0, 0,             1, 0}};

const double lidar_cloud_time = 0.1; // seconds or ts units
const int lidar_num_lasers = 64;
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
    if (ptr == map.end()) return ptr->second;
    ts d1 = abs(frame - ptr->first);
    ts d2 = abs(frame - ptr2->first);
    return d1 < d2 ? ptr->second : ptr2->second;
}
// Get transform from lidar frame to global frame
Eigen::Matrix4d T_lidar_global(pose_p pose) {
    return *pose * T_vehicle_lidar;
}

// Get transform from global frame to image pixel
vector<Point> project(Cloud::Ptr cloud, pose_p pose) {
    vector<Point> pixels;
    auto extrinsic = T_vehicle_camera_left.inverse() * pose->inverse();
    for(auto pt : cloud->points) {
        Eigen::Vector4d v = pt.getVector4fMap().cast<double>();
        v = extrinsic * v;
        if(v[3] == 0) continue; // point at infinity
        if(v[2] / v[3] < 0) continue; // point behind camera
        if(v[0] / v[3] < 0 || v[0] / v[3] > 2 * cx) continue; // out of frame
        if(v[1] / v[3] < 0 || v[1] / v[3] > 2 * cy) continue;
        
        Eigen::Vector3d pixel = T_projection * v;
        pixels.push_back(Point(pixel[0]/pixel[2], pixel[1]/pixel[2]));
    }
}

void getGroundPlane(
        Cloud::Ptr in_cloud, 
        Cloud::Ptr ground_cloud,
        Cloud::Ptr stuff_cloud
        ) {
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

// Clean and transform lidar cloud into global frame with dewarp
void processLidar(Cloud::Ptr lidar, ts frame) {
    cerr << "Processing lidar... ";
    // filter out shitty points
    Cloud::Ptr 
        temp_cloud(new Cloud),
        temp_cloud2(new Cloud);
    PointIndices::Ptr ind(new PointIndices);
    int n = lidar->size();
    for(int i=0; i<n; i++) {
        auto pt = lidar->at(i);
        if(pt.x < 0) continue; // remove points behind vehicle
        if(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z < lidar_near_sqr_thresh) {
            cerr << "c";
            continue;
        }
        ind->indices.push_back(i);
    }

    // transform cloud to desired locations
    auto ptr = poses.upper_bound(frame), 
         ptr2 = poses.lower_bound(frame + lidar_cloud_time);
    if(ptr == poses.begin() || ptr2 == poses.end()) {
        // no need to de-warp at the beginning
        if(ptr == poses.end()) ptr--;
        transformPointCloud(*lidar, ind->indices, *temp_cloud, T_lidar_global(ptr->second));
        lidar->swap(*temp_cloud);

        cerr << " lidar processed (begin or end)" << endl;
        return;
    }
    cerr << "dewarping... ";
    ptr--;
    auto t1 = ptr->first, t2 = ptr2->first;
    auto T1 = T_lidar_global(ptr->second),
         T2 = T_lidar_global(ptr2->second);
    assert(t1 <= frame && t2 > frame);
    transformPointCloud(*lidar, ind->indices, *temp_cloud, T1);
    transformPointCloud(*lidar, ind->indices, *temp_cloud2, T2);
    for(int i=0; i<temp_cloud->size(); i++) {
        // interpolate between transformed positions
        PointXYZ p1 = temp_cloud->at(i), p2 = temp_cloud2->at(i);
        ts point_time = frame +
            lidar_cloud_time * (ind->indices[i]/lidar_num_lasers) /
            (double) (n/lidar_num_lasers);
        ts s2 = (point_time - t1)/(t2 - t1),
           s1 = 1.0 - s2;
        lidar->at(i).x = p1.x * s1 + p2.x * s2;
        lidar->at(i).y = p1.y * s1 + p2.y * s2;
        lidar->at(i).z = p1.z * s1 + p2.z * s2;
    }

    cerr << "lidar processed" << endl;
}

// process the frame and do most of the work for this project
Mat processFrame(const deque<Mat> &camera_frames,
                 const deque<const_pose_p> &pose_frames,
                 const deque<Cloud::ConstPtr> &lidar_frames) {
    Mat out(camera_frames[0]);
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
    auto lidar_path_it = lidar_paths.begin();
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
        cout << "Current pose:\n" << *pose << endl;

        // read lidar
        while(lidar_path_it->first < frame) {
            auto lidar_path = lidar_path_it->second;
            Cloud::Ptr lidar(new Cloud);
            io::loadPCDFile<PointXYZ>(lidar_path.string(), *lidar);
            processLidar(lidar, lidar_path_it->first);
            lidar_path_it++;
            lidar_frames.push_front(lidar);
            while (lidar_frames.size() > num_frames_to_keep) lidar_frames.pop_back();
            cout << lidar->points.size() << " points loaded." << endl;
        }

        if(lidar_frames.size() < num_frames_to_keep) {
            // not enough lidar data to do anything, just show frame
            imshow(video, camera);
            continue;
        }

        // accumulate scans, downsample, and segment ground
        Cloud::Ptr lidar_aggregation(new Cloud);
        for(auto lidar : lidar_frames) {
            *lidar_aggregation += *lidar;
        }

        assert(lidar_aggregation.is_dense);

        Cloud::Ptr lidar_filtered(new Cloud);
        ApproximateVoxelGrid<PointXYZ> voxels;
        voxels.setInputCloud(lidar_aggregation);
        voxels.setLeafSize(voxel_size);
        voxels.filter(*lidar_filtered);

        Cloud::Ptr ground(new Cloud);
        Cloud::Ptr otherstuff(new Cloud);
        getGroundPlane(lidar_aggregation, ground, otherstuff);

        // process frame
        auto img = processFrame(camera_frames, pose_frames, lidar_frames);

        vector<Point> ground_pixels = project(ground, pose);
        vector<Point> other_pixels = project(otherstuff, pose);
        for (auto cvpt : ground_pixels) {
            circle(img, cvpt, 3, Scalar(0, 100, 200), 1, 8, 0);
        }
        for (auto cvpt : other_pixels) {
            circle(img, cvpt, 3, Scalar(255, 50, 0), 1, 8, 0);
        }

        // visualize
        imshow(video, img);
    }
    cvWaitKey();
    return 0;
}
