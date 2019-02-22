#ifndef _GRAPH_OPTIMIZE_H
#define _GRAPH_OPTIMIZE_H

#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <fstream>

namespace GRAPH_OPTIMIZE
{
	using namespace std;
        using namespace cv;

	class CGraphOptimize
	{
	public:
		static cv::Mat gray2binary(cv::Mat & gray);
		static cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);
                static void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode);
		//static cv::Mat delete_jut(cv::Mat& src, cv::Mat& dst, int uthreshold, int vthreshold, int type);
                static void imfillholes(cv::Mat &src);
		static void linear_fit0(int* x_pre, int* y_pre, int* x, int* y, int len);
		static void linear_fit1(int* x_pre, int* y_pre, int* x, int* y, int len);
                static void linear_fit(int* x_pre, int* y_pre, int* x, int* y, int len);
                static void linear_fit2(int* x_pre, int* y_pre, int* x, int* y, int len, int n);
                static cv::Mat Geo_area(int* data, int width, int height, int* path_x, int* path_y, int len, float x0, float y0, float f);
                static double angle(int x1, int y1, int x2, int y2, int x3, int y3);
                static void curve_out(int* x, int* y, int len, int l, int C);
                static cv::Mat Pathprocess(cv::Mat & src, int* path_x, int* path_y, int len, float x0, float y0, float f);
                static string Endpoint(cv::Mat & src, int* path_x, int* path_y, int len, float x0, float y0, float f);
                static cv::Mat Geo(int* data, int width, int height, int* path_x, int* path_y, int len, float x0, float y0, float f);
                static string getPathPoint();
	};

}

#endif

