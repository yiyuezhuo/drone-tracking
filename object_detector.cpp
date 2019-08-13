

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class ObjectDetector{
    public:
        Rect detect(Mat mat);
};

class HumanDetector{
    public:
        Rect detect(Mat mat);
};

Rect HumanDetector::detect(Mat mat){
    Rect InitBB = selectROI(mat, false);
    return InitBB;
}

class YOLODetector{
    public:
        Rect detect(Mat mat);
};

Rect YOLODetector::detect(Mat mat){
    return Rect(0,0,0,0);
}