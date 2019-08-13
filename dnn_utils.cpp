#include<vector>
#include<iostream>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
// It seems that opencv 3.4.2 doesn't require `#include <opencv2/dnn.hpp>` line

using namespace std;
using namespace cv;
using namespace cv::dnn; 

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
         
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
         
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, 
    vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes)
{
    /*
    // Export them
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    */
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
}

//int detect_dial_pointer(Net net, Mat frame, Rect& rect_dial, Rect& rect_pointer){ 
bool detect_drone(Net net, Mat frame, Rect& rect_drone){ 
    // frame is just a image mat, I am tired to rename them.
    Mat blob;
    vector<int> classIds; 
    vector<float> confidences;
    vector<Rect> boxes;

    // Create a 4D blob from a frame.
    blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
     
    //Sets the input to the network
    net.setInput(blob);
     
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    // Remove the bounding boxes with low confidence
    postprocess(frame, outs, classIds, confidences, boxes);

    // extract max
    int max_drone_idx = -1;
    float max_drone_conf = 0.0;

    for(size_t i=0; i<boxes.size(); i++){
        assert(classIds[i] == 0);
        if(confidences[i] > max_drone_conf){
            max_drone_conf = confidences[i];
            max_drone_idx = i;
        }
    }

    #ifdef DETECT_DEBUG
    
    if(max_dial_idx != -1)
        rectangle(frame, boxes[max_dial_idx], Scalar(255, 0, 0), 2);
    if(max_pointer_idx != -1)
        rectangle(frame, boxes[max_pointer_idx], Scalar(128, 0, 0), 2);

    #endif

    if(max_drone_idx == -1){
        cout << "fail to detect drone" << endl;
        return false; 
    }

    rect_drone = boxes[max_drone_idx];
    
    return true; // ok
}

