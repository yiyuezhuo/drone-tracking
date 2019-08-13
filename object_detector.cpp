
#include "dnn_utils.cpp"

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class ObjectDetector{
    public:
        Rect detect(Mat mat);
};

class HumanDetector{
    public:
        bool detect(Mat input, Rect& output);
};

class ObsessionHumandDector{
    // force select a ROI, disable fail returned, debug purpose. Assume a object is present.
    public:
        bool detect(Mat input, Rect& output);
};

bool HumanDetector::detect(Mat input, Rect& output){
    output = selectROI(input, false);
    cout << "selectROI" << output << " x:" << output.x << " y:" << output.y << " width:" << output.width << " height:" << output.height << endl;
    if ((output.x == 0) && (output.y ==0) && (output.width == 0) && (output.height == 0)){
        cout << "Human judge reject to select a roi. Object is missing or a mistake done from human." << endl;
        return false;
    }
    return true;
}

bool ObsessionHumandDector::detect(Mat input, Rect& output){
    bool ok = false;
    while(!ok){
        output = selectROI(input, false);
        cout << "selectROI" << output << " x:" << output.x << " y:" << output.y << " width:" << output.width << " height:" << output.height << endl;
        if ((output.x == 0) && (output.y ==0) && (output.width == 0) && (output.height == 0)){
            cout << "Human judge reject to select a roi. Do it again!" << endl;
            continue;
        }
        ok = true;
    }
    return true;
}

class YOLODetector{
    public:
        YOLODetector();
        bool YOLODetector::detect(Mat input, Rect& output);
    private:
        Net net;
};

YOLODetector::YOLODetector(){
    String modelConfiguration = "yolo-cfg/yolov3-tiny-bin.cfg";
    String modelWeights = "videos_frames/backup/yolov3-tiny-bin_obj_final.weights";

    net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    cout << "model load load succ from: " << modelWeights << endl;
}
bool YOLODetector::detect(Mat input, Rect& output){
    //return Rect(0,0,0,0);

    bool ok = detect_drone(net, input, output);

    Mat yolo_tracking_backup = input.clone();
    rectangle(yolo_tracking_backup, output, Scalar(255, 0, 0), 2);
    imshow("YOLO tracking", yolo_tracking_backup);
    cout << "YOLO tracking:" << output << endl;

    return ok;
}