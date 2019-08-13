/*
Here, we will define some detector which take history of tracker and determine if the anomaly occur.
*/

#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

class AnomalyDetector{
    public:
        bool step();
        bool stoped = false; // stop can be turn to false from true if you know what you are doing
};

class RectSizeDetector{
    public:
        bool step(float width, float height);
        bool stoped = false;
    private:
        float prev_width = -1;
        float prev_height = -1;
        bool running = false;
};

class RectSizeAndJumpDetector{
    public:
        bool step(float center_x, float center_y, float width, float height);
        bool stoped = false;
    private:
        float prev_center_x = -1;
        float prev_center_y = -1;
        float prev_width = -1;
        float prev_height = -1;
        bool running = false;
};


bool RectSizeDetector::step(float width, float height){
    assert(!stoped);

    cout << "AD prev_width:" << prev_width << " prev_height:" << prev_height << " width:" << width << " height:" << height << endl;

    if(!running){
        //cout << "p1" << endl;
        running = true;
        prev_width = width;
        prev_height = height;
        cout << "AD start running, pass current frame" << endl;
        return false;
    }
    else{
        
        float odd1 = (float)width / prev_width;
        float odd2 = (float)height / prev_height;
        
        cout << "test::" << odd1 << ", " << odd2 << ", " << ((odd1 > 2) || (odd1 <0.5) || (odd2 >2) || (odd2 < 0.5)) << endl;
        if ((odd1 > 2) || (odd1 <0.5) || (odd2 >2) || (odd2 < 0.5)){
            //cout << "p2" << endl;
            // reject to save.
            stoped = true; // caller should create another detector to run following dectetion
            //return true;
        }
        //cout << "p3" << endl;
        prev_width = width;
        prev_height = height;
        return stoped; // stoped == anomaly occur
        //return false;
    }
}


bool RectSizeAndJumpDetector::step(float center_x, float center_y, float width, float height){
    assert(!stoped);

    cout << "AD"  << " prev_center_x:" << prev_center_x << " prev_center_y:" << prev_center_y
        << " center_x:" << center_x << " center_y:" << center_y
        << " prev_width:" << prev_width << " prev_height:" << prev_height 
        << " width:" << width << " height:" << height << endl;

    if(!running){
        //cout << "p1" << endl;
        running = true;

        prev_center_x = center_x;
        prev_center_y = center_y;
        prev_width = width;
        prev_height = height;

        cout << "AD start running, pass current frame" << endl;
        return false;
    }
    else{
        
        float odd1 = (float)width / prev_width;
        float odd2 = (float)height / prev_height;

        bool odds_fail = (odd1 > 2) || (odd1 <0.5) || (odd2 >2) || (odd2 < 0.5);

        bool jump_fail = (abs(center_x - prev_center_x) > (width+ prev_width)) || (abs(center_y - prev_center_y) > (height+ prev_height));
        
        cout << "test::" << odd1 << ", " << odd2 << ", " << odds_fail << ", " << jump_fail << endl;
        
        if (odds_fail || jump_fail){
            //cout << "p2" << endl;
            // reject to save.
            stoped = true; // caller should create another detector to run following dectetion
            //return true;
        }
        //cout << "p3" << endl;

        prev_center_x = center_x;
        prev_center_y = center_y;
        prev_width = width;
        prev_height = height;

        return stoped; // stoped == anomaly occur
        //return false;
    }
}
