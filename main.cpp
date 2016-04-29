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
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
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
typedef PointCloud<PointXYZ> Cloud;
typedef shared_ptr<Eigen::Matrix4d> pose_p;
typedef shared_ptr<const Eigen::Matrix4d> const_pose_p;
typedef PointCloud<PointXYZ> Cloud;
namespace cv {
bool operator<(const Point a, const Point b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }
}

/************************* GLOBAL VARIABLES ************************/
// Configuration
const int num_frames_to_keep = 20;
const int min_lidar_frames_needed = 5;
const int ransac_iterations = 100;
const float voxel_size = 0.1; // m or lidar units
const int histogram_size = 30,
          histogram_offset = 40;
const int normalization_patch_radius = 2, // pixels
          min_colour_samples = 15,
          specularity_colour_threshold = 10,
          specular_within_threshold_limit = 10;

// thresholds
const double lidar_near_sqr_thresh = 4, // m^2
             furthest_point_y = 15, // m in vehicle frame
             ground_min_height = -3, // m in vehicle frame
             ground_max_height = -0.5, // m in vehicle frame
             ground_distance_thresh = 0.2, // m
             ground_neighbourhood = 30; // m^2

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

const ts lidar_cloud_time = 0.1; // seconds or ts units
const int lidar_num_lasers = 64;
// poses
map<ts, pose_p> poses;
/*********************** END GLOBAL VARIABLES **********************/

// string to timestamp
ts stots(string s) {
    return stod(s);
}
void loadData() {
    for(auto i = directory_iterator(path("data")), end = directory_iterator(); i != end; i++) {
        path file = i->path();
        string filename = file.filename().string();
        if(boost::starts_with(filename, "left")) {
            left_img_paths[stots(filename.substr(11, 20))] = file;
        } else if(boost::starts_with(filename, "right")) {
            right_img_paths[stots(filename.substr(12, 20))] = file;
        } else if(boost::starts_with(filename, "lidar")) {
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
    cerr << "Data load success!" << endl;
}
template<typename T> T getClosestFrame(ts frame, map<ts, T> &map) {
    auto ptr = map.upper_bound(frame);
    if(ptr == map.begin()) return ptr->second;
    auto ptr2 = ptr--;
    if(ptr2 == map.end()) return ptr->second;
    ts d1 = abs(frame - ptr->first);
    ts d2 = abs(frame - ptr2->first);
    return d1 < d2 ? ptr->second : ptr2->second;
}
// Get transform from lidar frame to global frame
Eigen::Matrix4d T_global_lidar(pose_p pose) {
    return *pose * T_vehicle_lidar;
}

// Get transform from camera frame to image pixel
vector<Point> project(Cloud::ConstPtr cloud,
                      const_pose_p pose,
                      vector<int> &valid_indices) {
    vector<Point> pixels;
    auto projection = T_projection * T_vehicle_camera_left.inverse();
    for(int i=0; i<cloud->size(); i++) {
        auto pt = cloud->at(i);
        Eigen::Vector3d pixel = projection * pt.getVector4fMap().cast<double>();
        Point p(pixel[0]/pixel[2], pixel[1]/pixel[2]);
        if(p.x < 0 || p.x > 2 * cx || p.y < 0 || p.y > 2 * cy) {
            // outside of frame
            continue;
        }
        valid_indices.push_back(i);
        pixels.push_back(p);
    }
    return pixels;
}
void getGroundPlane(Cloud::ConstPtr in_cloud,
                    const_pose_p pose,
                    Cloud::Ptr ground_cloud,
                    Cloud::Ptr stuff_cloud) {
    vector<int> filtered;
    PassThrough<PointXYZ> pass;
    pass.setInputCloud(in_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(ground_min_height, ground_max_height);
    pass.filter(filtered);

    // extract unfiltered indices into stuff_cloud
    Cloud::Ptr ground_in(new Cloud);
    ExtractIndices<PointXYZ> extract;
    extract.setInputCloud(in_cloud);
    extract.setIndices(boost::make_shared<vector<int>>(filtered));
    extract.setNegative(false);
    extract.filter(*ground_in);
    extract.setNegative(true);
    extract.filter(*stuff_cloud);


    ModelCoefficients coefficients;
    PointIndices::Ptr inliers(new PointIndices);
    // segment it!
    SACSegmentation<PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(SACMODEL_PLANE);
    seg.setMethodType(SAC_RANSAC);
    seg.setDistanceThreshold(ground_distance_thresh);
    seg.setMaxIterations(ransac_iterations);
    seg.setInputCloud(ground_in);
    seg.segment(*inliers, coefficients);
    //cerr << coefficients << endl;

    // extract the plane into a new point cloud
    Cloud::Ptr ground_plane(new Cloud);
    Cloud::Ptr tmp(new Cloud);
    ExtractIndices<PointXYZ> extract2;
    extract2.setInputCloud(ground_in);
    extract2.setIndices(inliers);
    extract2.setNegative(false);
    extract2.filter(*ground_plane);
    extract2.setNegative(true);
    extract2.filter(*tmp);
    *stuff_cloud += *tmp;

    // project the ground inliers
    ProjectInliers<PointXYZ> proj;
    proj.setModelType(SACMODEL_PLANE);
    proj.setInputCloud(ground_plane);
    proj.setModelCoefficients(boost::make_shared<ModelCoefficients>(coefficients));
    proj.filter(*ground_cloud);
/*
    double pose_z = (*pose)(2,3);
    cerr << pose_z << endl;
    vector<int> histogram(histogram_size, 0);
    for(auto p : tmp->points) {
        double x = p.x - (*pose)(0,3),
               y = p.y - (*pose)(1,3),
               z = p.z - (*pose)(2,3);
        if(x*x + y*y + z*z > ground_neighbourhood) continue;
        int zz = round((p.z - pose_z)/ground_distance_thresh) + histogram_offset;
        if(zz >= 0 && zz < histogram_size) histogram[zz]++;
    }
    int hist_max = 0, mode = 0;
    for(int i=0; i<histogram_size; i++) {
        if(histogram[i] > hist_max) {
            hist_max = histogram[i];
            mode = i;
        }
    }
    double ground_height = (mode - histogram_offset) * ground_distance_thresh;
    cerr << ground_height << endl;
    for(auto p : tmp->points) {
        if(abs(p.z - pose_z - ground_height) < ground_distance_thresh) {
            ground_cloud->push_back(p);
        } else {
            stuff_cloud->push_back(p);
        }
    }
*/
}
// Clean and transform lidar cloud into global frame with dewarp
void processLidar(Cloud::Ptr lidar, ts frame) {
    Cloud::Ptr lidar_unfiltered(new Cloud(*lidar));
    VoxelGrid<PointXYZ> voxels;
    voxels.setInputCloud(lidar_unfiltered);
    voxels.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxels.filter(*lidar);
    // filter out bad points
    PointIndices::Ptr ind(new PointIndices);
    Cloud::Ptr temp_cloud(new Cloud);
    int n = lidar->size();
    for(int i=0; i<n; i++) {
        auto pt = lidar->at(i);
        if(pt.x < 0) continue;
        if(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z < lidar_near_sqr_thresh) {
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
        transformPointCloud(*lidar, ind->indices, *temp_cloud, T_global_lidar(ptr->second));
        lidar.swap(temp_cloud);
        return;
    }
    Cloud::Ptr temp_cloud2(new Cloud);
    ptr--;
    auto t1 = ptr->first, t2 = ptr2->first;
    auto T1 = T_global_lidar(ptr->second),
         T2 = T_global_lidar(ptr2->second);
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
}
Mat processFrame(const deque<Mat> &camera_frames,
                 const deque<const_pose_p> &pose_frames,
                 const deque<Cloud::ConstPtr> &lidar_frames) {
    // accumulate scans, downsample, and segment ground
    Cloud::Ptr lidar_aggregation(new Cloud);
    for(auto lidar : lidar_frames) {
        *lidar_aggregation += *lidar;
    }

    Cloud::Ptr lidar_filtered(new Cloud);
    VoxelGrid<PointXYZ> voxels;
    voxels.setInputCloud(lidar_aggregation);
    voxels.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxels.filter(*lidar_filtered);

    auto pose = pose_frames[0];

    // Transform point to current frame
    lidar_aggregation.swap(lidar_filtered);
    lidar_filtered->clear();
    auto T = pose->inverse();
    for(int i=0; i<lidar_aggregation->size(); i++) {
        auto pt = lidar_aggregation->at(i);
        Eigen::Vector4d v = T * pt.getVector4fMap().cast<double>();
        if(v[3] == 0) continue; // point at infinity
        if(v[1] / v[3] < 0) continue; // point behind camera
        if(v[1] / v[3] > furthest_point_y) continue; // point too far ahead
        pt.x = v[0] / v[3];
        pt.y = v[1] / v[3];
        pt.z = v[2] / v[3];
        lidar_filtered->push_back(pt);
    }

    Cloud::Ptr ground(new Cloud), otherstuff(new Cloud);
    getGroundPlane(lidar_filtered, pose, ground, otherstuff);
    vector<int> valid_indices;
    auto ground_pixels = project(ground, pose, valid_indices);

    Mat out;
    cvtColor(camera_frames[0], out, COLOR_GRAY2BGR);
    vector<int> valid_indices_other;
    auto other_pixels = project(otherstuff, pose, valid_indices_other);
    for (int i=0; i<other_pixels.size(); i++) {
        //circle(out, other_pixels[i], 3, Scalar(255, 255, 0), 1, 8, 0);
    }

    map<Point, vector<uchar>> ground_colours;
    for(int i=camera_frames.size()-1; i>=0; i--) {
        vector<int> valid_indices;
        auto ground_pixels2 = project(ground, pose_frames[i], valid_indices);
        for(int j=0; j<valid_indices.size(); j++) {
            if (ground_pixels2[j].x < normalization_patch_radius || ground_pixels2[j].x + normalization_patch_radius >= out.cols ||
                ground_pixels2[j].y < normalization_patch_radius || ground_pixels2[j].y + normalization_patch_radius >= out.rows) {
                continue;
            }
            auto colour = camera_frames[i].at<uchar>(ground_pixels2[j]);
            double sum = 0;
            for (int a=-normalization_patch_radius; a <= normalization_patch_radius; a++) {
                for (int b=-normalization_patch_radius; b <= normalization_patch_radius; b++) {
                    double s = camera_frames[i].at<uchar>(ground_pixels2[j].y + b, ground_pixels2[j].x + a);
                    sum += s*s;
                }
            }
            colour /= sqrt(sum);
            ground_colours[ground_pixels[valid_indices[j]]].push_back(colour);
        }
    }
    for(auto p : ground_colours) {
        vector<uchar> colours(p.second.begin(), p.second.end());
        if (colours.size() < min_colour_samples) continue;
        sort(colours.begin(), colours.end());
        int front = 0, back = 0, mx = 1;
        while(front < colours.size()) {
          while(back < front && colours[front] - colours[back] > specularity_colour_threshold) back++;
          mx = max(mx, front-back+1);
          front++;
        }
        if (mx <= specular_within_threshold_limit) {
          cout << mx << endl;
          circle(out, p.first, 3, Scalar(0, 255, 0), 1, 8, 0);
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
    auto lidar_path_it = lidar_paths.begin();
    for(auto p : left_img_paths) {
        // frame timing
        ts frame = p.first;
        auto current_time = chrono::high_resolution_clock::now();
        if(last_frame_timestamp != -1) {
            auto timestamp_diff = frame - last_frame_timestamp;
            auto time_diff = chrono::duration<double>(current_time - last_frame_time).count();
            auto wait = max(1., (timestamp_diff - time_diff) * 1000);
            cvWaitKey(wait);
        }
        last_frame_timestamp = frame;
        last_frame_time = chrono::high_resolution_clock::now();
        cerr << fixed << "Current frame: " << frame << endl;

        // read camera image
        auto camera = imread(left_img_paths[frame].string(), 0);
        camera_frames.emplace_front(camera);
        if(camera_frames.size() > num_frames_to_keep) camera_frames.pop_back();

        // read pose
        auto pose = getClosestFrame(frame, poses);
        pose_frames.push_front(pose);
        if(pose_frames.size() > num_frames_to_keep) pose_frames.pop_back();
        //cerr << "Current pose: " << endl << *pose << endl;

        // read lidar
        while(lidar_path_it != lidar_paths.end() && lidar_path_it->first < frame) {
            auto lidar_path = lidar_path_it->second;
            Cloud::Ptr lidar(new Cloud);
            io::loadPCDFile<PointXYZ>(lidar_path.string(), *lidar);
            processLidar(lidar, lidar_path_it->first);
            lidar_frames.push_front(lidar);
            while(lidar_frames.size() > min_lidar_frames_needed) lidar_frames.pop_back();
            lidar_path_it++;
        }

        if(lidar_frames.size() < min_lidar_frames_needed) {
            // not enough lidar data to do anything, just show frame
            imshow(video, camera);
        } else {
            // process frame
            auto img = processFrame(camera_frames, pose_frames, lidar_frames);
            imshow(video, img);
        }
    }
    cvWaitKey();
    return 0;
}
