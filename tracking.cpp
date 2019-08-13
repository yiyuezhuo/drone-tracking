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
	//imshow("mMBSMap before", mMBSMap);
	normalize(mMBSMap, mMBSMap, 0.0, 1.0, NORM_MINMAX);
	//imshow("mMBSMap after", mMBSMap);
	mSaliencyMap = mMBSMap;
}

Mat MBS::getSaliencyMap()
{
	Mat ret;
	//imshow("mSaliencyMap", mSaliencyMap);
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
		//imshow("mDst", mDst);

		dst += mDst;
	}

	//imshow("dst", dst);
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

	assert(remove_border==false);
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

	//imshow("dst", dst);
	//imshow("srcRoi", srcRoi);

	MBS mbs(srcRoi, dst, state);
	mbs.computeSaliency(use_MBSPlus);
	Mat resultRoi = mbs.getSaliencyMap();
	Mat result = Mat::zeros(src_small.size(), CV_32FC1);

	//imshow("resultRoi", resultRoi);

	//cout << "roi:" << roi << " x:" << roi.x << " y:" << roi.y << " w:" << roi.width << " h:" << roi.height << endl;
	normalize(resultRoi, Mat(result, roi), 0.0, 1.0, NORM_MINMAX);

	return result;
}