// debug tool 

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/*
Type reference table
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


void inspect(Mat mat, string label){
    double min, max;
    minMaxLoc(mat, &min, &max);
    int count = countNonZero(mat);
    cout << label <<" rows:" << mat.rows << " cols:" << mat.cols << " type:"<< mat.type()
     << " min:" << min << " max:" << max << " #nonzero:" << count << endl;
}

template <class T>
void show_mat(Mat mat){
    for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            cout << mat.at< T >(i,j) << ", ";
        }
        cout << endl;
    }
    cout << endl;
}