/*
lab:
cl old.cpp opencv_world346.lib -ID:\opencv\opencv-3.4.6\build\include /link /OUT:"test.exe" /SUBSYSTEM:CONSOLE /MACHINE:X64 /LIBPATH:D:\opencv\opencv-3.4.6\build\x64\vc14\lib
pc:
cl old.cpp opencv_world342.lib -IE:\agent2\opencv-release-342\opencv\build\include /link /OUT:"test.exe" /SUBSYSTEM:CONSOLE /MACHINE:X64 /LIBPATH:E:\agent2\opencv-release-342\opencv\build\x64\vc14\lib
*/

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

static cv::RNG MBS_RNG;

class MBS
{
public:
	MBS(const cv::Mat& src);
	MBS(const cv::Mat& src, cv::Mat& mDst, cv::Mat& state);
	cv::Mat getSaliencyMap();
	void computeSaliency(bool use_MBSPlus = false);
	cv::Mat getMBSMap() const { return mMBSMap; }
private:
	cv::Mat mSaliencyMap;
	cv::Mat mMBSMap;
	int mAttMapCount;
	cv::Mat mBorderPriorMap;
	cv::Mat mSrc;
	cv::Mat mDst;
	cv::Mat KF_state;
	std::vector<cv::Mat> mFeatureMaps;
	void whitenFeatMap(float reg);
	void computeBorderPriorMap(float reg, float marginRatio);
};

cv::Mat computeCWS(const cv::Mat src, float reg, float marginRatio);
cv::Mat fastMBS(const std::vector<cv::Mat> featureMaps);
cv::Mat fastMBSPlus(const std::vector<cv::Mat> featureMaps, cv::Mat& mDst, cv::Mat& KF_state);
int findFrameMargin(const cv::Mat& img, bool reverse);
bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi);

#define MAX_IMG_DIM 600
#define TOLERANCE 0.01
#define FRAME_MAX 20
#define SOBEL_THRESH 0.4
#define EXPIXEL 25 // increase the value to get larger search area
MBS::MBS(const Mat& src)
	:mAttMapCount(0)
{
	mSrc = src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);

	split(mSrc, mFeatureMaps);

	for (int i = 0; i < mFeatureMaps.size(); i++)
	{
		//normalize(mFeatureMaps[i], mFeatureMaps[i], 255.0, 0.0, NORM_MINMAX);
		medianBlur(mFeatureMaps[i], mFeatureMaps[i], 5);
	}
}
MBS::MBS(const Mat& src, Mat& dst, Mat& state)
	:mAttMapCount(0)
{
	mSrc = src.clone();
	mDst = dst.clone();
	KF_state = state.clone();

	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
	assert(mDst.type() == CV_32FC1);

	split(mSrc, mFeatureMaps);

	for (int i = 0; i < mFeatureMaps.size(); i++)
	{
		//normalize(mFeatureMaps[i], mFeatureMaps[i], 255.0, 0.0, NORM_MINMAX);
		medianBlur(mFeatureMaps[i], mFeatureMaps[i], 5);
	}
}

void MBS::computeSaliency(bool use_MBSPlus)
{
	if (use_MBSPlus)
	{
		mMBSMap = fastMBSPlus(mFeatureMaps, mDst, KF_state);
	}
	else
	{
		mMBSMap = fastMBS(mFeatureMaps);
	}
	normalize(mMBSMap, mMBSMap, 0.0, 1.0, NORM_MINMAX);
	mSaliencyMap = mMBSMap;
}

Mat MBS::getSaliencyMap()
{
	Mat ret;
	normalize(mSaliencyMap, ret, 0.0, 255.0, NORM_MINMAX);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}


void rasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)
{
	Size sz = featMap.size();
	float *pMapup = (float*)map.data + 1;
	float *pMap = pMapup + sz.width;
	uchar *pFeatup = featMap.data + 1;
	uchar *pFeat = pFeatup + sz.width;
	uchar *pLBup = lb.data + 1;
	uchar *pLB = pLBup + sz.width;
	uchar *pUBup = ub.data + 1;
	uchar *pUB = pUBup + sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float lfV, upV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap - 1);
		featPrev = *(pFeat - 1);
		lbPrev = *(pLB - 1);
		ubPrev = *(pUB - 1);


		for (int c = 1; c < sz.width - 1; c++)
		{
			lfV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			upV = MAX(*pFeat, *pUBup) - MIN(*pFeat, *pLBup);//(*pFeat >= *pLBup && *pFeat <= *pUBup) ? *pMapup : *pMapup + abs((float)(*pFeat) - (float)(*pFeatup));

			flag = 0;
			if (lfV < *pMap)
			{
				*pMap = lfV;
				flag = 1;
			}
			if (upV < *pMap)
			{
				*pMap = upV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from left
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from up
				*pLB = MIN(*pFeat, *pLBup);
				*pUB = MAX(*pFeat, *pUBup);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap++; pMapup++;
			featPrev = *pFeat;
			pFeat++; pFeatup++;
			lbPrev = *pLB;
			pLB++; pLBup++;
			ubPrev = *pUB;
			pUB++; pUBup++;
		}
		pMapup += 2; pMap += 2;
		pFeat += 2; pFeatup += 2;
		pLBup += 2; pLB += 2;
		pUBup += 2; pUB += 2;
	}
}

void invRasterScan(const Mat& featMap, Mat& map, Mat& lb, Mat& ub)
{
	Size sz = featMap.size();
	int datalen = sz.width*sz.height;
	float *pMapdn = (float*)map.data + datalen - 2;
	float *pMap = pMapdn - sz.width;
	uchar *pFeatdn = featMap.data + datalen - 2;
	uchar *pFeat = pFeatdn - sz.width;
	uchar *pLBdn = lb.data + datalen - 2;
	uchar *pLB = pLBdn - sz.width;
	uchar *pUBdn = ub.data + datalen - 2;
	uchar *pUB = pUBdn - sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float rtV, dnV;
	int flag;
	for (int r = 1; r < sz.height - 1; r++)
	{
		mapPrev = *(pMap + 1);
		featPrev = *(pFeat + 1);
		lbPrev = *(pLB + 1);
		ubPrev = *(pUB + 1);

		for (int c = 1; c < sz.width - 1; c++)
		{
			rtV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			dnV = MAX(*pFeat, *pUBdn) - MIN(*pFeat, *pLBdn);//(*pFeat >= *pLBdn && *pFeat <= *pUBdn) ? *pMapdn : *pMapdn + abs((float)(*pFeat) - (float)(*pFeatdn));

			flag = 0;
			if (rtV < *pMap)
			{
				*pMap = rtV;
				flag = 1;
			}
			if (dnV < *pMap)
			{
				*pMap = dnV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from right
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from down
				*pLB = MIN(*pFeat, *pLBdn);
				*pUB = MAX(*pFeat, *pUBdn);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap--; pMapdn--;
			featPrev = *pFeat;
			pFeat--; pFeatdn--;
			lbPrev = *pLB;
			pLB--; pLBdn--;
			ubPrev = *pUB;
			pUB--; pUBdn--;
		}


		pMapdn -= 2; pMap -= 2;
		pFeatdn -= 2; pFeat -= 2;
		pLBdn -= 2; pLB -= 2;
		pUBdn -= 2; pUB -= 2;
	}
}

cv::Mat fastMBS(const std::vector<cv::Mat> featureMaps)
{
	assert(featureMaps[0].type() == CV_8UC1);

	Size sz = featureMaps[0].size();
	Mat ret = Mat::zeros(sz, CV_32FC1);
	if (sz.width < 3 || sz.height < 3)
		return ret;

	for (int i = 0; i < featureMaps.size(); i++)
	{
		Mat map = Mat::zeros(sz, CV_32FC1);
		Mat mapROI(map, Rect(1, 1, sz.width - 2, sz.height - 2));

		mapROI.setTo(Scalar(100000));

		Mat lb = featureMaps[i].clone();
		Mat ub = featureMaps[i].clone();

		rasterScan(featureMaps[i], map, lb, ub);
		invRasterScan(featureMaps[i], map, lb, ub);
		rasterScan(featureMaps[i], map, lb, ub);
		ret += map;
	}

	return ret;
}
// MBSPlusKF starts here: rasterScanPlus
void rasterScanPlus(const Mat& featMap, Mat& mDst, Mat& lb, Mat& ub, Rect& ExpBB)
{
	Point Alfup, Crtdn;
	Size sz = featMap.size();
	Alfup.x = ExpBB.x;
	Alfup.y = ExpBB.y;
	Crtdn.x = ExpBB.x + ExpBB.width - 1;
	Crtdn.y = ExpBB.y + ExpBB.height - 1;

	int Step_pointer = sz.width - ExpBB.width;

	float *pMapup = (float*)mDst.data + (ExpBB.y - 1)*sz.width + ExpBB.x;
	float *pMap = pMapup + sz.width;
	uchar *pFeatup = featMap.data + (ExpBB.y - 1)*sz.width + ExpBB.x;
	uchar *pFeat = pFeatup + sz.width;
	uchar *pLBup = lb.data + (ExpBB.y - 1)*sz.width + ExpBB.x;
	uchar *pLB = pLBup + sz.width;
	uchar *pUBup = ub.data + (ExpBB.y - 1)*sz.width + ExpBB.x;
	uchar *pUB = pUBup + sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float lfV, upV;
	int flag;
	for (int r = Alfup.y; r <= Crtdn.y; r++)
	{
		mapPrev = *(pMap - 1);
		featPrev = *(pFeat - 1);
		lbPrev = *(pLB - 1);
		ubPrev = *(pUB - 1);

		for (int c = Alfup.x; c <= Crtdn.x; c++)
		{
			lfV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			upV = MAX(*pFeat, *pUBup) - MIN(*pFeat, *pLBup);//(*pFeat >= *pLBup && *pFeat <= *pUBup) ? *pMapup : *pMapup + abs((float)(*pFeat) - (float)(*pFeatup));

			flag = 0;
			if (lfV < *pMap)
			{
				*pMap = lfV;
				flag = 1;
			}
			if (upV < *pMap)
			{
				*pMap = upV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from left
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from up
				*pLB = MIN(*pFeat, *pLBup);
				*pUB = MAX(*pFeat, *pUBup);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap++; pMapup++;
			featPrev = *pFeat;
			pFeat++; pFeatup++;
			lbPrev = *pLB;
			pLB++; pLBup++;
			ubPrev = *pUB;
			pUB++; pUBup++;
		}
		pMapup += Step_pointer; pMap += Step_pointer;
		pFeat += Step_pointer; pFeatup += Step_pointer;
		pLBup += Step_pointer; pLB += Step_pointer;
		pUBup += Step_pointer; pUB += Step_pointer;
	}
}
// MBSPlusKF: invRasterScanPlus
void invRasterScanPlus(const Mat& featMap, Mat& mDst, Mat& lb, Mat& ub, Rect& ExpBB)
{

	Point Alfup, Crtdn;
	Size sz = featMap.size();
	Alfup.x = ExpBB.x;
	Alfup.y = ExpBB.y;
	Crtdn.x = ExpBB.x + ExpBB.width - 1;
	Crtdn.y = ExpBB.y + ExpBB.height - 1;
	int Step_pointer = sz.width - ExpBB.width;

	float *pMapdn = (float*)mDst.data + (Crtdn.y + 1)*sz.width + Crtdn.x;
	float *pMap = pMapdn - sz.width;
	uchar *pFeatdn = featMap.data + (Crtdn.y + 1)*sz.width + Crtdn.x;
	uchar *pFeat = pFeatdn - sz.width;
	uchar *pLBdn = lb.data + (Crtdn.y + 1)*sz.width + Crtdn.x;
	uchar *pLB = pLBdn - sz.width;
	uchar *pUBdn = ub.data + (Crtdn.y + 1)*sz.width + Crtdn.x;
	uchar *pUB = pUBdn - sz.width;

	float mapPrev;
	float featPrev;
	uchar lbPrev, ubPrev;

	float rtV, dnV;
	int flag;
	for (int r = Alfup.y; r <= Crtdn.y; r++)
	{
		mapPrev = *(pMap + 1);
		featPrev = *(pFeat + 1);
		lbPrev = *(pLB + 1);
		ubPrev = *(pUB + 1);

		for (int c = Alfup.x; c <= Crtdn.x; c++)
		{
			rtV = MAX(*pFeat, ubPrev) - MIN(*pFeat, lbPrev);//(*pFeat >= lbPrev && *pFeat <= ubPrev) ? mapPrev : mapPrev + abs((float)(*pFeat) - featPrev);
			dnV = MAX(*pFeat, *pUBdn) - MIN(*pFeat, *pLBdn);//(*pFeat >= *pLBdn && *pFeat <= *pUBdn) ? *pMapdn : *pMapdn + abs((float)(*pFeat) - (float)(*pFeatdn));

			flag = 0;
			if (rtV < *pMap)
			{
				*pMap = rtV;
				flag = 1;
			}
			if (dnV < *pMap)
			{
				*pMap = dnV;
				flag = 2;
			}

			switch (flag)
			{
			case 0:		// no update
				break;
			case 1:		// update from right
				*pLB = MIN(*pFeat, lbPrev);
				*pUB = MAX(*pFeat, ubPrev);
				break;
			case 2:		// update from down
				*pLB = MIN(*pFeat, *pLBdn);
				*pUB = MAX(*pFeat, *pUBdn);
				break;
			default:
				break;
			}

			mapPrev = *pMap;
			pMap--; pMapdn--;
			featPrev = *pFeat;
			pFeat--; pFeatdn--;
			lbPrev = *pLB;
			pLB--; pLBdn--;
			ubPrev = *pUB;
			pUB--; pUBdn--;
		}

		pMapdn -= Step_pointer; pMap -= Step_pointer;
		pFeatdn -= Step_pointer; pFeat -= Step_pointer;
		pLBdn -= Step_pointer; pLB -= Step_pointer;
		pUBdn -= Step_pointer; pUB -= Step_pointer;
	}
}
// MBSPlusKF: fastMBSPlus
cv::Mat fastMBSPlus(const std::vector<cv::Mat> featureMaps, Mat& mDst, Mat& KF_state)
{
	assert(featureMaps[0].type() == CV_8UC1);
	assert(mDst.type() == CV_32FC1);

	Size sz = featureMaps[0].size();
	Mat dst = Mat::zeros(sz, CV_32FC1);

	Rect KF_InitBB;
	KF_InitBB.width = KF_state.at<float>(4);
	KF_InitBB.height = KF_state.at<float>(5);
	KF_InitBB.x = KF_state.at<float>(0) - 0.5*KF_InitBB.width;
	KF_InitBB.y = KF_state.at<float>(1) - 0.5*KF_InitBB.height;

	Rect ExpBB;
	ExpBB.x = max(KF_InitBB.x - EXPIXEL, 1);
	ExpBB.y = max(KF_InitBB.y - EXPIXEL, 1);
	int ExpBBxRight = min(ExpBB.x + 2 * EXPIXEL + KF_InitBB.width - 1, sz.width - 1);
	int ExpBByBottom = min(ExpBB.y + 2 * EXPIXEL + KF_InitBB.height - 1, sz.height - 1);
	ExpBB.width = ExpBBxRight - ExpBB.x;
	ExpBB.height = ExpBByBottom - ExpBB.y;

	for (int i = 0; i < featureMaps.size(); i++)
	{
		Mat mDstROI(mDst, Rect(ExpBB));
		mDstROI.setTo(Scalar(100000));
		Mat lb = featureMaps[i].clone();
		Mat ub = featureMaps[i].clone();

		rasterScanPlus(featureMaps[i], mDst, lb, ub, ExpBB);
		invRasterScanPlus(featureMaps[i], mDst, lb, ub, ExpBB);
		rasterScanPlus(featureMaps[i], mDst, lb, ub, ExpBB);

		dst += mDst;
	}

	return dst;
}

int findFrameMargin(const Mat& img, bool reverse)
{
	Mat edgeMap, edgeMapDil, edgeMask;
	Sobel(img, edgeMap, CV_16SC1, 0, 1);
	edgeMap = abs(edgeMap);
	edgeMap.convertTo(edgeMap, CV_8UC1);
	edgeMask = edgeMap < (SOBEL_THRESH * 255.0);
	dilate(edgeMap, edgeMapDil, Mat(), Point(-1, -1), 2);
	edgeMap = edgeMap == edgeMapDil;
	edgeMap.setTo(Scalar(0.0), edgeMask);

	if (!reverse)
	{
		for (int i = edgeMap.rows - 1; i >= 0; i--)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return i + 1;
	}
	else
	{
		for (int i = 0; i < edgeMap.rows; i++)
			if (mean(edgeMap.row(i))[0] > 0.6*255.0)
				return edgeMap.rows - i;
	}

	return 0;
}

bool removeFrame(const cv::Mat& inImg, cv::Mat& outImg, cv::Rect &roi)
{
	if (inImg.rows < 2 * (FRAME_MAX + 3) || inImg.cols < 2 * (FRAME_MAX + 3))
	{
		roi = Rect(0, 0, inImg.cols, inImg.rows);
		outImg = inImg;
		return false;
	}

	Mat imgGray;
	cvtColor(inImg, imgGray, CV_RGB2GRAY);

	int up, dn, lf, rt;

	up = findFrameMargin(imgGray.rowRange(0, FRAME_MAX), false);
	dn = findFrameMargin(imgGray.rowRange(imgGray.rows - FRAME_MAX, imgGray.rows), true);
	lf = findFrameMargin(imgGray.colRange(0, FRAME_MAX).t(), false);
	rt = findFrameMargin(imgGray.colRange(imgGray.cols - FRAME_MAX, imgGray.cols).t(), true);

	int margin = MAX(up, MAX(dn, MAX(lf, rt)));
	if (margin == 0)
	{
		roi = Rect(0, 0, imgGray.cols, imgGray.rows);
		outImg = inImg;
		return false;
	}

	int count = 0;
	count = up == 0 ? count : count + 1;
	count = dn == 0 ? count : count + 1;
	count = lf == 0 ? count : count + 1;
	count = rt == 0 ? count : count + 1;

	// cut four border region if at least 2 border frames are detected
	if (count > 1)
	{
		margin += 2;
		roi = Rect(margin, margin, inImg.cols - 2 * margin, inImg.rows - 2 * margin);
		outImg = Mat(inImg, roi);

		return true;
	}

	// otherwise, cut only one border
	up = up == 0 ? up : up + 2;
	dn = dn == 0 ? dn : dn + 2;
	lf = lf == 0 ? lf : lf + 2;
	rt = rt == 0 ? rt : rt + 2;


	roi = Rect(lf, up, inImg.cols - lf - rt, inImg.rows - up - dn);
	outImg = Mat(inImg, roi);

	return true;

}

Mat doWork(
	const Mat& src,
	bool use_Gray,
	bool remove_border,
	bool use_MBSPlus
)
{
	Mat src_small;
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w, h);
	resize(src, src_small, Size((int)(MAX_IMG_DIM*w / maxD), (int)(MAX_IMG_DIM*h / maxD)), 0.0, 0.0, INTER_AREA);// standard: width: 300 pixel
	Mat srcRoi;
	Rect roi;
	// detect and remove the artifical frame of the image
	if (remove_border)
		removeFrame(src_small, srcRoi, roi);
	else
	{
		srcRoi = src_small;
		roi = Rect(0, 0, src_small.cols, src_small.rows);
	}

	if (use_Gray)
		cvtColor(srcRoi, srcRoi, CV_RGB2GRAY);

	/* Computing saliency */
	MBS mbs(srcRoi);
	mbs.computeSaliency(use_MBSPlus);

	Mat resultRoi = mbs.getSaliencyMap();
	Mat result = Mat::zeros(src_small.size(), CV_32FC1);

	normalize(resultRoi, Mat(result, roi), 0.0, 1.0, NORM_MINMAX);
	return result;
}
// MBSPlusKF: doWorkPlus
Mat doWorkPlus(
	const Mat& src,
	Mat& dst,
	Mat& state,
	bool use_Gray,
	bool remove_border,
	bool use_MBSPlus
)
{
	Mat src_small;
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w, h);

	resize(src, src_small, Size((int)(MAX_IMG_DIM*w / maxD), (int)(MAX_IMG_DIM*h / maxD)), 0.0, 0.0, INTER_AREA);// standard: width: 300 pixel
	Mat srcRoi;
	Rect roi;

	// detect and remove the artifical frame of the image
	if (remove_border)
		removeFrame(src_small, srcRoi, roi);
	else
	{
		srcRoi = src_small;
		roi = Rect(0, 0, src_small.cols, src_small.rows);
	}

	if (use_Gray)
		cvtColor(srcRoi, srcRoi, CV_RGB2GRAY);

	MBS mbs(srcRoi, dst, state);
	mbs.computeSaliency(use_MBSPlus);
	Mat resultRoi = mbs.getSaliencyMap();
	Mat result = Mat::zeros(src_small.size(), CV_32FC1);

	normalize(resultRoi, Mat(result, roi), 0.0, 1.0, NORM_MINMAX);

	return result;
}


int main(int argc, char **argv)
{

	Mat src, img, img1, image, img_dst;
	double TotalTime = 0.0, AveTime = 0.0;
	//char imgInPath[256], imgOutPath[256], imgOutPath1[256], imgOutPath2[256], BboxPath[256];
	string imgpath, TrackingResPath;
	//std::vector<cv::String> filenames;
	//cv::String folder;

	imgpath = "videos/5.avi";

	//imgpath = "D:/���˻�����/Tracking/4.avi";
	VideoCapture video(imgpath);
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		return 1;
	}
		

		
	bool use_Gray = true;
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
	//��������ڣ������������ģ�ͽ��м�⣬��һ֡��⵽�����������к������ٹ���!!
	//!!!!!!!!!!!!!!!!!!!!!���û�м�⵽�������һֱ���!!!!!!!!!!!!!!!!!!!!!
	//!!!!!!!!!!!!!!!!!!!!!��ⷶΧ������ͼ��!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//##############################################//
	Rect InitBB = selectROI(img, false);

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
	
	long int i = 0;	
	while (video.read(src))
	{
		////src = imread(imgInPath, 1);

		// Start the clock timing
		clock_t begin_t = clock();
		if (i % 20 == 0) // update the complete saliency map every ten frames
		{
			use_MBSPlus = false;
			img_dst = doWork(src, use_Gray, remove_border, use_MBSPlus);
			img = img_dst.clone();
			use_MBSPlus = true;
		}

		// kalman filter 
		resize(src, src, Size((int)(MAX_IMG_DIM*w2 / maxD2), (int)(MAX_IMG_DIM*h2 / maxD2)), 0.0, 0.0, INTER_AREA);
		img_dst = doWorkPlus(src, img, state, use_Gray, remove_border, use_MBSPlus);

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
		Rect InitBB = boundingRect(Mat(points));
		////outBB << ImageScale*InitBB.x << "," << ImageScale*InitBB.y << "," << ImageScale*InitBB.width << "," << ImageScale*InitBB.height << endl;
		//##############################################//
		//�����жϣ�������ľ��ο�䶯̫����Ҫ���м�����������
		//һ�����ڵ�ǰĿ�긽���ķ�Χ�ڼ��
		int ww = state.at<float>(4);
		int hh = state.at<float>(5);
		if (InitBB.width / ww > 2 || InitBB.height / hh > 2)
		{
			cout << "@@@@@@@@@@@@@ToolargeMeasurement@@@@@@@@@@@@@";
			//��ʱ�����Ե�ǰ״̬�����ĵ�crop��Ȼ�����ѧϰ���!!
			//��ȡλ�ô�С���ٸ���״̬!!update InitBB
			//InitBB.x = -1;
		}
			
		//##############################################//
		cv::rectangle(src, InitBB, CV_RGB(0, 255, 0), 2);

		// Kalman Update
		if (InitBB.x < 0)
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
			ToolargeMeasurement = 0;

			meas.at<float>(0) = InitBB.x + InitBB.width / 2;
			meas.at<float>(1) = InitBB.y + InitBB.height / 2;
			meas.at<float>(2) = (float)InitBB.width;
			meas.at<float>(3) = (float)InitBB.height;

			if (!found) // First detection!
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
				//��һ���ֿ����޸ģ��Ҳ���Ŀ���������ѧϰģ�ͽ���Ŀ���⣡
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
		}
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
	
	
	system("pause");
	return 0;
}