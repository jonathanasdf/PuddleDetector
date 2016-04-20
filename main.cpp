#include <cassert>
#include <iostream>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace boost::filesystem;
using namespace cv;
using namespace pcl;
using namespace std;
typedef long double ts;
typedef map<string, path> DataMap;
/************************* GLOBAL VARIABLES ************************/
// Data paths
DataMap left_img_paths, right_img_paths, lidar_paths;
/*********************** END GLOBAL VARIABLES **********************/
void loadData() {
    path datadir("data");
    for (auto i = directory_iterator(datadir), end = directory_iterator(); i != end; i++) {
        path file = i->path();
        string filename = file.filename().string();
        if (boost::starts_with(filename, "left")) {
            left_img_paths[filename.substr(11, 20)] = file;
        } else if (boost::starts_with(filename, "right")) {
            right_img_paths[filename.substr(12, 20)] = file;
        } else if (boost::starts_with(filename, "lidar")) {
            lidar_paths[filename.substr(12, 20)] = file;
        }
    }
    cout << "Data load success!" << endl;
}
path getClosestFrame(string frame, DataMap &map) {
    auto ptr2 = map.lower_bound(frame);
    auto ptr = ptr2++;
    if (ptr2 == map.end()) return ptr->second;
    ts timestamp = stold(frame);
    ts d1 = abs(timestamp - stold(ptr->first));
    ts d2 = abs(timestamp - stold(ptr2->first));
    return d1 < d2 ? ptr->second : ptr2->second;
}
int main(int argc, char **argv) {
    std::cout << "Hello world!" << std::endl;
    loadData();

    char video[] = "video";
    cvNamedWindow(video);
    pcl::visualization::CloudViewer pcl_viewer("pointcloud");

    ts last_frame_time = -1;
    for(auto p : left_img_paths) {
        string frame = p.first;
        if (!right_img_paths.count(frame)) continue;

        ts timestamp = stold(frame);
        if (last_frame_time != -1) {
          cvWaitKey((timestamp - last_frame_time) * 1000);
        }
        last_frame_time = timestamp;

        auto left = imread(left_img_paths[frame].string());
        auto right = imread(left_img_paths[frame].string());
        assert(left.rows == right.rows);
        assert(left.type() == right.type());
        Mat combined(left.rows, left.cols + right.cols, left.type());
        left.copyTo(combined(Rect(0, 0, left.cols, left.rows)));
        right.copyTo(combined(Rect(left.cols, 0, right.cols, right.rows)));
        imshow(video, combined);

        auto lidar_path = getClosestFrame(frame, lidar_paths);
        pcl::PointCloud<pcl::PointXYZ>::Ptr lidar (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ> (lidar_path.string(), *lidar);
        pcl_viewer.showCloud(lidar);
    }
    cvWaitKey();
    return 0;
}
