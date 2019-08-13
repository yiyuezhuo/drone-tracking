/*
lab:
cl old.cpp opencv_world346.lib -ID:\opencv\opencv-3.4.6\build\include /link /OUT:"test.exe" /SUBSYSTEM:CONSOLE /MACHINE:X64 /LIBPATH:D:\opencv\opencv-3.4.6\build\x64\vc14\lib
pc:
cl old.cpp opencv_world342.lib -IE:\agent2\opencv-release-342\opencv\build\include /link /OUT:"test.exe" /SUBSYSTEM:CONSOLE /MACHINE:X64 /LIBPATH:E:\agent2\opencv-release-342\opencv\build\x64\vc14\lib
*/

#include "config.hpp"
#include "utils.cpp"

#include "anomaly_detector.cpp"
#include "object_detector.cpp"
#include "tracking.cpp"

int main(int argc, char **argv)
{
	auto anomaly_detector = RectSizeDetector();
	auto object_detector = HumanDetector();

	Mat src, img, img1, image, img_dst;
	double TotalTime = 0.0, AveTime = 0.0;
	//char imgInPath[256], imgOutPath[256], imgOutPath1[256], imgOutPath2[256], BboxPath[256];
	string imgpath, TrackingResPath;
	//std::vector<cv::String> filenames;
	//cv::String folder;

	//imgpath = "videos/5.avi";
	imgpath = VIDEO_PATH;

	VideoCapture video(imgpath);
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		return 1;
	}
		

	#ifdef USE_RGB
	bool use_Gray = false;
	#else
	bool use_Gray = true;
	#endif
	bool remove_border = false;
	bool use_MBSPlus = false;
							// get dMap1 of the 1st frame using fastMBS, doWork
	//Mat img1;
	bool ok = video.read(img1);

	imshow("Tracking", img1);
	waitKey(1);
	
	img = doWork(img1, use_Gray, remove_border, use_MBSPlus);

	// uncomment to write the result to TrackingResPath:
	// normalize(img, image, 0.0, 255.0, NORM_MINMAX);
	// image.convertTo(image, CV_8UC1);
	// imwrite(imgOutPath1, image);

	// starting of Kalman Filter for 1st frame
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;

	unsigned int type = CV_32F;
	cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

	cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
	cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

										// Transition State Matrix A
	cv::setIdentity(kf.transitionMatrix);
	kf.transitionMatrix.at<float>(2) = 1.0 / 30;
	kf.transitionMatrix.at<float>(9) = 1.0 / 30;

	// Measure Matrix H
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
	bool found = false;
	int ToolargeMeasurement = 0;
	// start of detection (measurement)
	Size framesz = img.size();
	Mat frame = Mat::zeros(framesz, CV_32FC1);
	frame = img.clone();
	normalize(frame, frame, 0.0, 255.0, NORM_MINMAX);
	frame.convertTo(frame, CV_8UC1);
	assert(frame.type() == CV_8UC1);

	threshold(frame, frame, 200, 255, THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(5, 3));
	dilate(frame, frame, element);

	std::vector<cv::Point> points;
	cv::Mat_<uchar>::iterator it = frame.begin<uchar>();
	cv::Mat_<uchar>::iterator end = frame.end<uchar>();
	for (; it != end; ++it)
	{
		if (*it)
		{
			points.push_back(it.pos());
		}
	}
	// draw the InitBB
	//Rect InitBB = boundingRect(Mat(points));
	//##############################################//
	//在最初环节，可以利用深度模型进行检测，第一帧检测到，则立即进行后续跟踪过程!!
	//!!!!!!!!!!!!!!!!!!!!!如果没有检测到，则可以一直检测!!!!!!!!!!!!!!!!!!!!!
	//!!!!!!!!!!!!!!!!!!!!!检测范围是整幅图像!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//##############################################//
	
	//Rect InitBB = selectROI(img, false);
	Rect InitBB = object_detector.detect(img);

	float w2 = (float)img1.cols, h2 = (float)img1.rows;
	float maxD2 = max(w2, h2);
	float ImageScale = maxD2 / MAX_IMG_DIM;

	resize(img1, img1, Size((int)(MAX_IMG_DIM*w2 / maxD2), (int)(MAX_IMG_DIM*h2 / maxD2)), 0.0, 0.0, INTER_AREA);

	meas.at<float>(0) = InitBB.x + InitBB.width / 2;
	meas.at<float>(1) = InitBB.y + InitBB.height / 2;
	meas.at<float>(2) = (float)InitBB.width;
	meas.at<float>(3) = (float)InitBB.height;

	kf.errorCovPre.at<float>(0) = 1;
	kf.errorCovPre.at<float>(7) = 1;
	kf.errorCovPre.at<float>(14) = 1;
	kf.errorCovPre.at<float>(21) = 1;
	kf.errorCovPre.at<float>(28) = 1;
	kf.errorCovPre.at<float>(35) = 1;

	kf.errorCovPost.at<float>(0) = 1;
	kf.errorCovPost.at<float>(7) = 1;
	kf.errorCovPost.at<float>(14) = 1;
	kf.errorCovPost.at<float>(21) = 1;
	kf.errorCovPost.at<float>(28) = 1;
	kf.errorCovPost.at<float>(35) = 1;

	state.at<float>(0) = meas.at<float>(0);
	state.at<float>(1) = meas.at<float>(1);
	state.at<float>(2) = 0;
	state.at<float>(3) = 0;
	state.at<float>(4) = meas.at<float>(2);
	state.at<float>(5) = meas.at<float>(3);

	state.copyTo(kf.statePost);
	found = true;
	// end of Kalman Filter for 1st frame

	rectangle(img1, InitBB, Scalar(0, 255, 0), 2, 1);
	imshow("Tracking", img1);
	waitKey(1);
	// Starting MBSPlusKF: fastMBSPlus, doWorkPlus
	use_MBSPlus = true;
	Rect predRect, CorrectRect;

	#ifdef BACKGROUND_MODELLING
	auto backSub = createBackgroundSubtractorMOG2();
	Mat fgMask;
	#endif

	//cout << "before object created" << endl;


	//cout << "object created" << endl;
	
	long int i = 0;	
	while (video.read(src))
	{
		//cout << "start new iteration" << endl;
		//src = imread(imgInPath, 1);

		// Start the clock timing
		clock_t begin_t = clock();
		if (i % FRAME_TO_GLOBAL == 0) // update the complete saliency map every ten frames
		{
			use_MBSPlus = false;
			img_dst = doWork(src, use_Gray, remove_border, use_MBSPlus);
			img = img_dst.clone();
			use_MBSPlus = true;
		}

		//cout << "it doWork" << endl; 
		 
		resize(src, src, Size((int)(MAX_IMG_DIM*w2 / maxD2), (int)(MAX_IMG_DIM*h2 / maxD2)), 0.0, 0.0, INTER_AREA);
		
		#ifdef BACKGROUND_MODELLING
		backSub->apply(src, fgMask);
		//cout << "src.rows:" << src.rows << " src.cols:" << src.cols << endl;
		//cout << "fgMask.rows:" << fgMask.rows << " fgMask.cols:" << fgMask.cols << endl;
		//inspect(src, "src"); // for some reason, it doesn't work
		//inspect(fgMask, "fgMask");
		imshow("FG Mask", fgMask);
		#endif

		//cout << "before doWorkPlus" << endl;
		
		// kalman filter
		#ifdef DISABLE_LOCAL_SCAN
		use_MBSPlus = false;
		img_dst = doWork(src, use_Gray, remove_border, use_MBSPlus);
		use_MBSPlus = true;
		#else
		img_dst = doWorkPlus(src, img, state, use_Gray, remove_border, use_MBSPlus);
		#endif
		// uncomment to write the saliency map to TrackingResPath:
		// normalize(img_dst, image, 0.0, 255.0, NORM_MINMAX);
		// image.convertTo(image, CV_8UC1);
		// imwrite(imgOutPath1, image);		
		imshow("ttt", img_dst);
		waitKey(1);

		if (found)
		{
			state = kf.predict();

			predRect.width = state.at<float>(4);
			predRect.height = state.at<float>(5);
			predRect.x = state.at<float>(0) - predRect.width / 2;
			predRect.y = state.at<float>(1) - predRect.height / 2;
		}

		//cout << "after kf.predict" << endl;

		// start of detection (measurement)
		Size framesz = img_dst.size();
		Mat frame = Mat::zeros(framesz, CV_32FC1);
		frame = img_dst.clone();
		normalize(frame, frame, 0.0, 255.0, NORM_MINMAX);
		frame.convertTo(frame, CV_8UC1);
		assert(frame.type() == CV_8UC1);

		threshold(frame, frame, 200, 255, THRESH_BINARY);
		// adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV,5, 7);
		// adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV,5, 7);
		// bitwise_not(map, map);

		Mat element = getStructuringElement(MORPH_RECT, Size(5, 3));
		dilate(frame, frame, element);

		// uncomment to write the binary map to TrackingResPath:
		// normalize(frame, image, 0.0, 255.0, NORM_MINMAX);
		// image.convertTo(image, CV_8UC1);			
		// imwrite(imgOutPath2, image);

		#ifdef BACKGROUND_MODELLING

		double min, max;
		minMaxLoc(fgMask, &min, &max); // 0-255
		//cout << "FG Mask min:" << min <<" max:" << max << endl;

		inspect(frame, "frame");
		inspect(fgMask, "fgMask");
		#ifdef PAUSE_PER_FRAME
		waitKey();
		#endif

		Mat andFrame = frame.clone();
		Mat orFrame = frame.clone();
		if ((max == 255) && (min ==0)){ // skip first frame
			for(int i=0; i<frame.rows; i++){
				for(int j=0; j<frame.cols; j++){
					//cout << "i:" << i << " j:" << j << endl;
					if (fgMask.at<uchar>(i,j) == 0){
						andFrame.at<uchar>(i,j) = 0;
					}
					if(fgMask.at<uchar>(i,j) == 255){
						orFrame.at<uchar>(i,j) = 255;
					}
				}
			}
		}
		//cout << "iter end" << endl;
		imshow("and frame", andFrame);
		imshow("or frame", orFrame);
		//inspect(andFrame, "andFrame");
		//inspect(orFrame, "orFrame");
		waitKey(1);

		#endif

		std::vector<cv::Point> points;
		cv::Mat_<uchar>::iterator it = frame.begin<uchar>();
		cv::Mat_<uchar>::iterator end = frame.end<uchar>();
		for (; it != end; ++it)
		{
			if (*it)
			{
				points.push_back(it.pos());
			}
		}

		cout << "points.size():" << points.size() << endl;
		if(points.size() == 0){
			cout << "Empty points detected" << endl;
		}

		//##############################################//
		#ifdef TRACK_VARIANCE
		cout << "errorCovPre:" << endl;
		show_mat<float>(kf.errorCovPre);
		cout << "errorCovPost:" << endl;
		show_mat<float>(kf.errorCovPost);
		//Mat src_var = src.clone();
		#endif

		// draw the InitBB
		Rect InitBB = boundingRect(Mat(points));
		////outBB << ImageScale*InitBB.x << "," << ImageScale*InitBB.y << "," << ImageScale*InitBB.width << "," << ImageScale*InitBB.height << endl;
		//##############################################//
		//加入判断，如果检测的矩形框变动太大，需要进行检测纠正！！！
		//一方面在当前目标附近的范围内检测
		//inspect(state, "state"); // float type
		//float ww = state.at<float>(4);
		//float hh = state.at<float>(5);

		/*
		#ifdef TOO_LARGE_DROP_UPDATE
		bool tooLarge = false;
		#endif

		if (InitBB.width / ww > 2 || InitBB.height / hh > 2)
		{
			cout << "@@@@@@@@@@@@@ToolargeMeasurement@@@@@@@@@@@@@";
			//此时可以以当前状态的中心点crop，然后深度学习检测!!
			//获取位置大小后再更新状态!!update InitBB
			//InitBB.x = -1;
			#ifdef TOO_LARGE_DROP_UPDATE
			tooLarge = true;
			#endif
		}
		*/

		bool anomaly_detected = anomaly_detector.step(InitBB.width, InitBB.height);
		//bool anomaly_detected = false;

		if (anomaly_detected){
			cout << "anomaly occur" << endl;
			//found = false;  // trigger kf reset, is it useful?
			found = true;
			Rect InitBB = object_detector.detect(src);
			anomaly_detector = RectSizeDetector();
			//waitKey();
		}
			
		cv::rectangle(src, InitBB, CV_RGB(0, 255, 0), 2);

		// Kalman Update
		/*
		if ((InitBB.x < 0))
		{
			ToolargeMeasurement++;
			cout << "ToolargeMeasurement@@@@@@@@@@@@@:" << ToolargeMeasurement << endl;
			if (ToolargeMeasurement >= 100)
			{
				found = false;
			}
		}
		else
		{
		*/
			//ToolargeMeasurement = 0;

		meas.at<float>(0) = InitBB.x + InitBB.width / 2;
		meas.at<float>(1) = InitBB.y + InitBB.height / 2;
		meas.at<float>(2) = (float)InitBB.width;
		meas.at<float>(3) = (float)InitBB.height;

		if (!found) // First detection or resume from chaos
		{
			// Initialization
			kf.errorCovPre.at<float>(0) = 1; // px
			kf.errorCovPre.at<float>(7) = 1; // px
			kf.errorCovPre.at<float>(14) = 1;
			kf.errorCovPre.at<float>(21) = 1;
			kf.errorCovPre.at<float>(28) = 1; // px
			kf.errorCovPre.at<float>(35) = 1; // px

			state.at<float>(0) = meas.at<float>(0);
			state.at<float>(1) = meas.at<float>(1);
			state.at<float>(2) = 0;
			state.at<float>(3) = 0;
			state.at<float>(4) = meas.at<float>(2);
			state.at<float>(5) = meas.at<float>(3);

			found = true;
			//##############################################//
			//这一部分可以修改，找不到目标后采用深度学习模型进行目标检测！
			//##############################################//
		}
		else
		{
			state = kf.correct(meas); // Kalman Correction

			CorrectRect.width = kf.statePost.at<float>(4);
			CorrectRect.height = kf.statePost.at<float>(5);
			CorrectRect.x = kf.statePost.at<float>(0) - CorrectRect.width / 2;
			CorrectRect.y = kf.statePost.at<float>(1) - CorrectRect.height / 2;

			// print the Bbox for calculation of presicion: 
			//cout << ImageScale*CorrectRect.x << "," << ImageScale*CorrectRect.y << "," << 							ImageScale*CorrectRect.width << "," << ImageScale*CorrectRect.height << endl;
		}
		//}
		// End the clock timing
		clock_t end_t = clock();
		double timeSec = (end_t - begin_t) / static_cast <double>(CLOCKS_PER_SEC);
		TotalTime = TotalTime + timeSec;
		imshow("Tracking", src);
		i++;
		////imshow("tracking", src);
		waitKey(1);

		cout << "timeSec=" << timeSec << endl;
		// end of kalman filter
		////img = img_dst.clone();
		// uncomment to write the TrackingResPath
		// normalize(img_dst, img_dst, 0.0, 255.0, NORM_MINMAX);
		// img_dst.convertTo(img_dst, CV_8UC1);
		// float w0 = (float)img_dst.cols, h0 = (float)img_dst.rows;
		// resize(img_dst,img_dst,Size((int)(w0*854/MAX_IMG_DIM),(int)(h0*854/MAX_IMG_DIM)),0.0,0.0,INTER_AREA);
		// imwrite(imgOutPath, src);

	}
	AveTime = TotalTime / (i+1);
	double FPS = 1.0 / AveTime;

	////outBB.close();
	cout << "AveTime for " << i+1 << " images, using MBSPlusKF is: " << AveTime << " s." << endl;
	cout << "Frame rate for " << imgpath << " is: " << FPS << " fps." << endl;
	////cout << "It's the " << Img_seq[j] << endl;
	
	//system("pause");
	return 0;
}