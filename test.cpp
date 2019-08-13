#include "config.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <vector>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;
using namespace std;

/*
+--------+----+----+----+----+------+------+------+------+
|        | C1 | C2 | C3 | C4 | C(5) | C(6) | C(7) | C(8) |
+--------+----+----+----+----+------+------+------+------+
| CV_8U  |  0 |  8 | 16 | 24 |   32 |   40 |   48 |   56 |
| CV_8S  |  1 |  9 | 17 | 25 |   33 |   41 |   49 |   57 |
| CV_16U |  2 | 10 | 18 | 26 |   34 |   42 |   50 |   58 |
| CV_16S |  3 | 11 | 19 | 27 |   35 |   43 |   51 |   59 |
| CV_32S |  4 | 12 | 20 | 28 |   36 |   44 |   52 |   60 |
| CV_32F |  5 | 13 | 21 | 29 |   37 |   45 |   53 |   61 |
| CV_64F |  6 | 14 | 22 | 30 |   38 |   46 |   54 |   62 |
+--------+----+----+----+----+------+------+------+------+
 */

int main(){
    Mat frame = Mat::zeros(Size(600, 337), CV_8UC1);

    imshow("mframeat", frame);

    for(int i=0; i< frame.rows; i++){
        for(int j=0; j<frame.cols; j++){
            frame.at<uchar>(i,j) = 255;
        }
    }

    imshow("mat1p", frame);

    assert(frame.type() == CV_8UC1);
    cout << "frame.type():" << frame.type() << endl;
    cout << "CV_8UC1:" << CV_8UC1 << " CV_8UC3:" << CV_8UC3 << endl;
    //assert(frame.type() == CV_8UC3);

    //Mat mat2 = mat.clone();
    //imshow("mat2", mat2);
    
    cout << "end" << endl;
    waitKey();
    //system("pause");

    return 0;
}