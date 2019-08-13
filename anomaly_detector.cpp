/*
Here, we will define some detector which take history of tracker and determine if the anomaly occur.
*/

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class RectSizeDetector{
    public:
        bool step(float width, float height);
    private:
        float prev_width;
        float prev_height;
        bool running = false;
        bool stoped = false;
};

bool RectSizeDetector::step(float width, float height){
    assert(!stoped);

    if(!running){
        running = true;
        prev_width = width;
        prev_height = height;
        return false;
    }
    else{
        if ((width / prev_width >2) || (height / prev_height >2)){
            // reject to save.
            stoped = true; // caller should create another detector to run following dectetion
            return true;
        }
        prev_width = width;
        prev_height = height;
        return false;
    }
}