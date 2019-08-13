
//#define TooLargeDropUpdate

#define BackgroundModelling

#define VIDEO_PATH "videos/5.avi"

//#define DISABLE_LOCAL_SCAN

// debug tool 

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void inspect(Mat mat, string label){
    double min, max;
    minMaxLoc(mat, &min, &max);
    int count = countNonZero(mat);
    cout << label <<" rows:" << mat.rows << " cols:" << mat.cols << " type:"<< mat.type()
     << " min:" << min << "max:" << max << " #nonzero:" << count << endl;
}