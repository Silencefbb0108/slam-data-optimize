#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <math.h>
using namespace std;
using namespace cv;

#define MAXN 90000

#define PI 3.1415926535897932384626
#define mul 4

//去除二值图像边缘的突出部
//uthreshold、vthreshold分别表示突出部的宽度阈值和高度阈值
//type代表突出部的颜色，0表示黑色，1代表白色 
void delete_jut(cv::Mat& src, cv::Mat& dst, int uthreshold, int vthreshold, int type) {

    src.copyTo(dst);
    int height = dst.rows;
    int width = dst.cols;
    int k;  //用于循环计数传递到外部
    for (int i = 0; i < height - 1; i++)
    {
	uchar* p = dst.ptr<uchar>(i);
	for (int j = 0; j < width - 1; j++)
	{
	    if (type == 0)
	    {
		//行消除
		if (p[j] == 255 && p[j + 1] == 0)
		{
		    if (j + uthreshold >= width)
		    {
			for (k = j + 1; k < width; k++)
			    p[k] = 255;
		    }
		    else
		    {
			for (k = j + 2; k <= j + uthreshold; k++)
			{
			    if (p[k] == 255) break;
			}
			if (p[k] == 255)
			{
			    for (int h = j + 1; h < k; h++)
				p[h] = 255;
			}
		    }
		}
		//列消除
		if (p[j] == 255 && p[j + width] == 0)
		{
		    if (i + vthreshold >= height)
		    {
			for (k = j + width; k < j + (height - i)*width; k += width)
			    p[k] = 255;
		    }
		    else
		    {
			for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
			{
			    if (p[k] == 255) break;
			}
			if (p[k] == 255)
			{
			    for (int h = j + width; h < k; h += width)
				p[h] = 255;
			}
		    }
		}
	    }
	    else  //type = 1
	    {
		//行消除
		if (p[j] == 0 && p[j + 1] == 255)
		{
		    if (j + uthreshold >= width)
		    {
			for (k = j + 1; k < width; k++)
			    p[k] = 0;
		    }
		    else
		    {
			for (k = j + 2; k <= j + uthreshold; k++)
			{
			    if (p[k] == 0) break;
			}
			if (p[k] == 0)
			{
			    for (int h = j + 1; h < k; h++)
				p[h] = 0;
			}
		    }
		}
		//列消除
		if (p[j] == 0 && p[j + width] == 255)
		{
		    if (i + vthreshold >= height)
		    {
			for (k = j + width; k < j + (height - i)*width; k += width)
			    p[k] = 0;
		    }
		    else
		    {
			for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
			{
			    if (p[k] == 0) break;
			}
		        if (p[k] == 0)
			{
			    for (int h = j + width; h < k; h += width)
				p[h] = 0;
			}
		    }
		}
	    }
	}
    }

}

//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;  
void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)  
{     
    int RemoveCount = 0;       //记录除去的个数  
    //记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
    cv::Mat Pointlabel = cv::Mat::zeros( Src.size(), CV_8UC1 );  

    if(CheckMode == 1)  
    {  
        //cout<<"Mode: 去除小区域. ";
        for(int i = 0; i < Src.rows; ++i)
        {    
            uchar* iData = Src.ptr<uchar>(i);
            uchar* iLabel = Pointlabel.ptr<uchar>(i);
            for(int j = 0; j < Src.cols; ++j)
            {    
                if (iData[j] < 10)
                {    
                    iLabel[j] = 3;
                }    
            }    
        }    
    }  
    else  
    {  
        //cout<<"Mode: 去除孔洞. ";  
        for(int i = 0; i < Src.rows; ++i)    
        {    
            uchar* iData = Src.ptr<uchar>(i);  
            uchar* iLabel = Pointlabel.ptr<uchar>(i);  
            for(int j = 0; j < Src.cols; ++j)    
            {    
                if (iData[j] > 10)    
                {    
                    iLabel[j] = 3;   
                }    
            }    
        }    
    }  
  
    std::vector<cv::Point2i> NeihborPos;  //记录邻域点位置  
    NeihborPos.push_back(cv::Point2i(-1, 0));  
    NeihborPos.push_back(cv::Point2i(1, 0));  
    NeihborPos.push_back(cv::Point2i(0, -1));  
    NeihborPos.push_back(cv::Point2i(0, 1));  
    if (NeihborMode == 1)  
    {  
        //cout<<"Neighbor mode: 8邻域."<<endl;  
        NeihborPos.push_back(cv::Point2i(-1, -1));  
        NeihborPos.push_back(cv::Point2i(-1, 1));  
        NeihborPos.push_back(cv::Point2i(1, -1));  
        NeihborPos.push_back(cv::Point2i(1, 1));  
    }  
    //else cout<<"Neighbor mode: 4邻域."<<endl;  
    int NeihborCount = 4 + 4*NeihborMode;  
    int CurrX = 0, CurrY = 0;  
    //开始检测  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 0)    
            {    
                //********开始该点处的检查**********  
                std::vector<cv::Point2i> GrowBuffer; //堆栈，用于存储生长点  
                GrowBuffer.push_back( cv::Point2i(j, i) );  
                Pointlabel.at<uchar>(i, j) = 1;  
                int CheckResult=0; //用于判断结果（是否超出大小），0为未超出，1为超出  
  
                for ( int z = 0; z < GrowBuffer.size(); z++ )  
                {  
  
                    for (int q = 0; q < NeihborCount; q++) //检查四个邻域点  
                    {  
                        CurrX = GrowBuffer.at(z).x+NeihborPos.at(q).x;  
                        CurrY = GrowBuffer.at(z).y+NeihborPos.at(q).y;  
                        if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows) //防止越界  
                        {  
                            if ( Pointlabel.at<uchar>(CurrY, CurrX) == 0 )  
                            {  
                                GrowBuffer.push_back( cv::Point2i(CurrX, CurrY) ); //邻域点加入buffer  
                                Pointlabel.at<uchar>(CurrY, CurrX) = 1; //更新邻域点的检查标签，避免重复检查  
                            }  
                        }  
                    }  
  
                }  
                if (GrowBuffer.size() > AreaLimit) CheckResult = 2; //判断结果（是否超出限定的大小），1为未超出，2为超出  
                else {CheckResult = 1; RemoveCount++;}  
                for (int z = 0; z < GrowBuffer.size(); z++) //更新Label记录  
                {  
                    CurrX = GrowBuffer.at(z).x;   
                    CurrY = GrowBuffer.at(z).y;  
                    Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;  
                }  
                //********结束该点处的检查**********  
  
  
            }    
        }    
    }    
  
    CheckMode = 255*(1 - CheckMode);  
    //开始反转面积过小的区域  
    for(int i = 0; i < Src.rows; ++i)    
    {    
        uchar* iData = Src.ptr<uchar>(i);  
        uchar* iDstData = Dst.ptr<uchar>(i);  
        uchar* iLabel = Pointlabel.ptr<uchar>(i);  
        for(int j = 0; j < Src.cols; ++j)    
        {    
            if (iLabel[j] == 2)    
            {    
                iDstData[j] = CheckMode;   
            }    
            else if(iLabel[j] == 3)  
            {  
                iDstData[j] = iData[j];  
            }  
        }    
    }   
      
    //cout<<RemoveCount<<" objects removed."<<endl;  
}

void imfillholes(cv::Mat &src)
{
    // detect external contours
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    //提取最外轮廓，参数contours中的每个轮廓用该轮廓的所有顶点表示;
    //src为8位单通道二值图像
    findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); 
    // fill external contours
    if( !contours.empty() && !hierarchy.empty() )
    {
        for (int idx = 0; idx < contours.size(); idx++)
        {
            drawContours(src, contours, idx, cv::Scalar::all(255), CV_FILLED, 8); //采用8-连通填充轮廓
        }
    }
    contours.clear();
    hierarchy.clear();
}

vector<cv::Point2i> getRotatePoint(cv::Mat srcImage, vector<cv::Point2i> Points, const cv::Point rotate_center, const double angle) {
	vector<cv::Point2i> dstPoints;
	int x1 = 0, y1 = 0;
	int row = srcImage.rows;
	for (size_t i = 0; i < Points.size(); i++)
	{
		x1 = Points.at(i).x;
		y1 = row - Points.at(i).y;
		int x2 = rotate_center.x;
		int y2 = row - rotate_center.y;
		int x = cvRound((x1 - x2)*cos(PI / 180.0 * angle) - (y1 - y2)*sin(PI / 180.0 * angle) + x2);
		int y = cvRound((x1 - x2)*sin(PI / 180.0 * angle) + (y1 - y2)*cos(PI / 180.0 * angle) + y2);
		y = row - y;
		dstPoints.push_back(Point2i(x, y));
	}
	return dstPoints;
}

cv::Mat Geo_area(int* data, int width, int height, int* path_x, int* path_y, int len, float x0, float y0, float f) {

    cv::Mat src = cv::Mat::zeros(height, width, CV_8UC1);
    for(int i = 0; i < src.rows; ++i) {
        for(int j = 0; j < src.cols; ++j) {
            src.at<uchar>(i, j) = data[width*i+j];
        }
    }
    
    cv::Mat src2 = cv::Mat::zeros(src.size(), CV_8UC1);
    src2 = (80 <= src & src <= 100);
    //imshow("src2", src2);

    if(countNonZero(src2) == 0) {

       cv::Mat zoom;
       cv::resize(src, zoom, cv::Size(), mul*f, mul*f, cv::INTER_NEAREST);
       cv::medianBlur(zoom, zoom, 9);
       cv::medianBlur(zoom, zoom, 5);
       cv::medianBlur(zoom, zoom, 3);

       cv::Mat color_map = cv::Mat::zeros(zoom.size(), CV_8UC3);

       for(int i = 0; i < zoom.rows; ++i) {
           for(int j = 0; j < zoom.cols; ++j) {
               if(zoom.at<uchar>(i, j) < 80) { //墙内
                  color_map.at<cv::Vec3b>(i, j)[0] = 245;
	          color_map.at<cv::Vec3b>(i, j)[1] = 81;
	          color_map.at<cv::Vec3b>(i, j)[2] = 116;
               }
               else if(zoom.at<uchar>(i, j) > 100) { //墙外
                  color_map.at<cv::Vec3b>(i, j)[0] = 241;
	          color_map.at<cv::Vec3b>(i, j)[1] = 119;
	          color_map.at<cv::Vec3b>(i, j)[2] = 160;
               }
               else { //墙体
                  color_map.at<cv::Vec3b>(i, j)[0] = 247;
	          color_map.at<cv::Vec3b>(i, j)[1] = 167;
	          color_map.at<cv::Vec3b>(i, j)[2] = 205;
               }
           }
       }
       cv::blur(color_map, color_map, cv::Size(3,3));

       return color_map;
    } 

    else {

	cv::Mat src3 = cv::Mat::zeros(src.size(), CV_8UC1);
	threshold(src, src3, 79, 255, cv::THRESH_BINARY);

	src3 = ~src3; //墙内为255，墙体和墙外为0

	cv::Mat labels2, stats2, centroids2;
	int nccomps2 = cv::connectedComponentsWithStats(src3, labels2, stats2, centroids2, 4); //连通域分析，src3为8位单通道二值图像

	int x_row2[len], y_col2[len];
	int x_pix_row2[len], y_pix_col2[len];

	memset(x_row2, 0, sizeof(x_row2));
	memset(y_col2, 0, sizeof(y_col2));
	memset(x_pix_row2, 0, sizeof(x_pix_row2));
	memset(y_pix_col2, 0, sizeof(y_pix_col2));

	int h_min2 = src.cols, h_max2 = 0, v_min2 = src.rows, v_max2 = 0;
	int h5 = 1, h6 = len-1, h7 = 1, h8 = len-1;
        for(int i = 0; i < len; ++i) {
            x_row2[i] = int(path_x[i]/5.0 - x0*20);
            y_col2[i] = int(path_y[i]/5.0 - y0*20);
            x_pix_row2[i] = src.cols - y_col2[i];
            y_pix_col2[i] = src.rows - x_row2[i];

            if(x_pix_row2[i] < h_min2 && i != 0) {h5 = i; h_min2 = x_pix_row2[i];}
            if(x_pix_row2[i] > h_max2 && i != 0) {h6 = i; h_max2 = x_pix_row2[i];}
            if(y_pix_col2[i] < v_min2 && i != 0) {h7 = i; v_min2 = y_pix_col2[i];}
            if(y_pix_col2[i] > v_max2 && i != 0) {h8 = i; v_max2 = y_pix_col2[i];}
        }

        cv::Mat blank_area = cv::Mat::zeros(src.size(), CV_8UC1);
        for(int i = 0; i < src.rows; i ++) {
            for(int j = 0; j < src.cols; j ++) {
                if(labels2.at<int>(i, j) == labels2.at<int>(y_pix_col2[h5], x_pix_row2[h5]) || 
                   labels2.at<int>(i, j) == labels2.at<int>(y_pix_col2[h6], x_pix_row2[h6]) || 
                   labels2.at<int>(i, j) == labels2.at<int>(y_pix_col2[h7], x_pix_row2[h7]) || 
                   labels2.at<int>(i, j) == labels2.at<int>(y_pix_col2[h8], x_pix_row2[h8])) {
                   blank_area.at<char>(i, j) = 255;
                }
            }
        }
        //imshow("blank_area", blank_area);

	cv::Mat known_area2;
	bitwise_or(blank_area, src2, known_area2);
        //imshow("known_area2", known_area2);
        cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        erode(known_area2, known_area2, element);
	dilate(known_area2, known_area2, element);
        //imshow("pre-fill", known_area2);

	imfillholes(known_area2);
        //imshow("post-fill", known_area2);
	src = ~src;
        //imshow("anti-src", src);
	bitwise_and(known_area2, src, known_area2);
        //imshow("post-and", known_area2);
	known_area2 = ~known_area2;
        //imshow("anti-known_area2", known_area2);

        cv::Mat B_known_area2 = cv::Mat::zeros(src.size(), CV_8UC1);
        B_known_area2 = (known_area2 < 80);

        cv::Mat labels3, stats3, centroids3;
	int nccomps3 = cv::connectedComponentsWithStats(B_known_area2, labels3, stats3, centroids3, 4); //连通域分析，src3为8位单通道二值图像

        cv::Mat blank_area2 = cv::Mat::zeros(src.size(), CV_8UC1);
        for(int i = 0; i < src.rows; i ++) {
            for(int j = 0; j < src.cols; j ++) {
                if(labels3.at<int>(i, j) == labels3.at<int>(y_pix_col2[h5], x_pix_row2[h5]) || 
                   labels3.at<int>(i, j) == labels3.at<int>(y_pix_col2[h6], x_pix_row2[h6]) || 
                   labels3.at<int>(i, j) == labels3.at<int>(y_pix_col2[h7], x_pix_row2[h7]) || 
                   labels3.at<int>(i, j) == labels3.at<int>(y_pix_col2[h8], x_pix_row2[h8])) {
                   blank_area2.at<char>(i, j) = 255;
                }
            }
        }

        cv::Mat BL_known_area2 = cv::Mat::zeros(src.size(), CV_8UC1);
        BL_known_area2 = (80 <= known_area2 & known_area2 <= 100);
        cv::Mat BLA_known_area2 = cv::Mat::zeros(src.size(), CV_8UC1);
        bitwise_or(blank_area2, BL_known_area2, BLA_known_area2);
	//cv::Mat element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        //erode(BLA_known_area2, BLA_known_area2, element2);
	//dilate(BLA_known_area2, BLA_known_area2, element2);
        cv::Mat BLAN_known_area2 = cv::Mat::zeros(src.size(), CV_8UC1);
        bitwise_and(BLA_known_area2, src, BLAN_known_area2);
	BLAN_known_area2 = ~BLAN_known_area2;

        std::vector<int> ms;
	for(int i = 1; i < src.rows-1; ++i) {
	    for(int j = 0; j < src.cols; ++j) {
		if((BLAN_known_area2.at<uchar>(i, j) > 100 || BLAN_known_area2.at<uchar>(i, j) < 80) && (BLAN_known_area2.at<uchar>(i+1, j) >= 80 && BLAN_known_area2.at<uchar>(i+1, j) <= 100)) {
		    ms.push_back(j);
		}  
	    }
	    if(ms.size() != 0) {
	       for(int k = 0; k < ms.size()-1; k ++) {
		   int cc = 0;
                   int dd = 0;
		   for(int t = ms[k]+1; t < ms[k+1]; t ++) {
		       if((BLAN_known_area2.at<uchar>(i, t) >= 80 && BLAN_known_area2.at<uchar>(i, t) <= 100) && (BLAN_known_area2.at<uchar>(i+1, t) >= 80 && BLAN_known_area2.at<uchar>(i+1, t) <= 100))
			   cc ++;
                       if(BLAN_known_area2.at<uchar>(i-1, t) >= 80 && BLAN_known_area2.at<uchar>(i-1, t) <= 100) { dd ++; }
	           }
		   if(cc == ms[k+1]-ms[k]-1 && dd == 0) {
		      for(int t = ms[k]+1; t < ms[k+1]; t ++) {
			  BLAN_known_area2.at<uchar>(i, t) = BLAN_known_area2.at<uchar>(i-1, t);
		      }
		   }  
	       }
	       ms.clear();
	    }
        }

        std::vector<int> ms2;
        for(int j = 1; j < src.cols-1; ++j) {
            for(int i = 0; i < src.rows; ++i) {
                if((BLAN_known_area2.at<uchar>(i, j) < 80 || BLAN_known_area2.at<uchar>(i, j) > 100) && (BLAN_known_area2.at<uchar>(i, j+1) >= 80 && BLAN_known_area2.at<uchar>(i, j+1) <= 100)) {
                   ms2.push_back(i);
                }  
            }
            if(ms2.size() != 0) {
	       for(int k = 0; k < ms2.size()-1; k ++) {
		   int cc = 0;
                   int dd = 0;
		   for(int t = ms2[k]+1; t < ms2[k+1]; t ++) {
		       if((BLAN_known_area2.at<uchar>(t, j) >= 80 && BLAN_known_area2.at<uchar>(t, j) <= 100) && (BLAN_known_area2.at<uchar>(t, j+1) >= 80 && BLAN_known_area2.at<uchar>(t, j+1) <= 100))
		          cc ++;
                       if(BLAN_known_area2.at<uchar>(t, j-1) >= 80 && BLAN_known_area2.at<uchar>(t, j-1) <= 100) { dd ++; }
		   }
		   if(cc == ms2[k+1]-ms2[k]-1 && dd == 0) {
		      for(int t = ms2[k]+1; t < ms2[k+1]; t ++) {
		          BLAN_known_area2.at<uchar>(t, j) = BLAN_known_area2.at<uchar>(t, j-1);
		      }
		   }
	       }
	       ms2.clear();
            }
        }

        std::vector<int> ms3;
        for(int i = src.rows-2; i > 0; --i) {
            for(int j = 0; j < src.cols; ++j) {
                if((BLAN_known_area2.at<uchar>(i, j) < 80 || BLAN_known_area2.at<uchar>(i, j) > 100) && (BLAN_known_area2.at<uchar>(i-1, j) >= 80 && BLAN_known_area2.at<uchar>(i-1, j) <= 100)) {
                   ms3.push_back(j);
                }  
            }
            if(ms3.size() != 0) {
	       for(int k = 0; k < ms3.size()-1; k ++) {
		   int cc = 0;
                   int dd = 0;
		   for(int t = ms3[k]+1; t < ms3[k+1]; t ++) {
		       if((BLAN_known_area2.at<uchar>(i, t) >= 80 && BLAN_known_area2.at<uchar>(i, t) <= 100) && (BLAN_known_area2.at<uchar>(i-1, t) >= 80 && BLAN_known_area2.at<uchar>(i-1, t) <= 100))
		          cc ++;
                       if(BLAN_known_area2.at<uchar>(i+1, t) >= 80 && BLAN_known_area2.at<uchar>(i+1, t) <= 100) { dd ++; }
		   }
		   if(cc == ms3[k+1]-ms3[k]-1 && dd == 0) {
		      for(int t = ms3[k]+1; t < ms3[k+1]; t ++) {
		          BLAN_known_area2.at<uchar>(i, t) = BLAN_known_area2.at<uchar>(i+1, t);
		      }
		   }
	       }
	       ms3.clear();
            }
        }

        std::vector<int> ms4;
        for(int j = src.cols-2; j > 0; --j) {
            for(int i = 0; i < src.rows; ++i) {
                if((BLAN_known_area2.at<uchar>(i, j) < 80 || BLAN_known_area2.at<uchar>(i, j) > 100) && (BLAN_known_area2.at<uchar>(i, j-1) >= 80 && BLAN_known_area2.at<uchar>(i, j-1) <= 100)) {
                   ms4.push_back(i);
                }  
            }
            if(ms4.size() != 0) {
	       for(int k = 0; k < ms4.size()-1; k ++) {
		   int cc = 0;
                   int dd = 0;
		   for(int t = ms4[k]+1; t < ms4[k+1]; t ++) {
		       if((BLAN_known_area2.at<uchar>(t, j) >= 80 && BLAN_known_area2.at<uchar>(t, j) <= 100) && (BLAN_known_area2.at<uchar>(t, j-1) >= 80 && BLAN_known_area2.at<uchar>(t, j-1) <= 100))
		          cc ++;
                       if(BLAN_known_area2.at<uchar>(t, j+1) >= 80 && BLAN_known_area2.at<uchar>(t, j+1) <= 100) { dd ++; }
		   }
		   if(cc == ms4[k+1]-ms4[k]-1 && dd == 0) {
		      for(int t = ms4[k]+1; t < ms4[k+1]; t ++) {
		          BLAN_known_area2.at<uchar>(t, j) = BLAN_known_area2.at<uchar>(t, j+1);
		      }
		   }
	       }
	       ms4.clear();
            }
        }
        //imshow("BLAN_known_area2", BLAN_known_area2);

        cv::Mat BLANK2 = cv::Mat::zeros(src.size(), CV_8UC1);
        BLANK2 = (BLAN_known_area2 < 80);
        cv::Mat BLANK3 = cv::Mat::zeros(src.size(), CV_8UC1);
        BLANK2.copyTo(BLANK3);
        //imshow("BLANK3", BLANK3);

        for(int i = 1; i < src.rows-1; ++i) {
            for(int j = 1; j < src.cols-1; ++j) {
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j-1) == 0) {
                   BLANK2.at<uchar>(i, j-1) = 255;
                }
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i-1, j) == 0) {
                   BLANK2.at<uchar>(i-1, j) = 255;
                }
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j+1) == 0) {
                   BLANK2.at<uchar>(i, j+1) = 255;
                }
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i+1, j) == 0) {
                   BLANK2.at<uchar>(i+1, j) = 255;
                }
                if((BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j-1) == 0) && (BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i-1, j) == 0)) {
                   BLANK2.at<uchar>(i-1, j-1) = 255;
                }
                if((BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j+1) == 0) && (BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i+1, j) == 0)) {
                   BLANK2.at<uchar>(i+1, j+1) = 255;
                }
                if((BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j-1) == 0) && (BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i+1, j) == 0)) {
                   BLANK2.at<uchar>(i+1, j-1) = 255;
                }
                if((BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i-1, j) == 0) && (BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j+1) == 0)) {
                   BLANK2.at<uchar>(i-1, j+1) = 255;
                }
            }
        }
        //imshow("BLANK2", BLANK2);

        cv::Mat BARRIER2 = cv::Mat::zeros(src.size(), CV_8UC1);
        BARRIER2 = (80 <= BLAN_known_area2 & BLAN_known_area2 <= 100);
        cv::Mat BARRIER3 = cv::Mat::zeros(src.size(), CV_8UC1);
        bitwise_and(BLANK2, BARRIER2, BARRIER3);
        //imshow("BARRIER3", BARRIER3);
        cv::Mat NOT_KNOWN0 = cv::Mat::zeros(src.size(), CV_8UC1);
        bitwise_or(BLANK3, BARRIER3, NOT_KNOWN0);
        //imshow("NOT_KNOWN0", NOT_KNOWN0);
	cv::Mat NOT_KNOWN = cv::Mat::zeros(src.size(), CV_8UC1);
	delete_jut(NOT_KNOWN0, NOT_KNOWN, 3, 3, 1);
        bitwise_and(NOT_KNOWN, ~BLAN_known_area2, NOT_KNOWN);
        //imshow("NOT_KNOWN2", NOT_KNOWN);
        //RemoveSmallRegion(NOT_KNOWN, NOT_KNOWN, 100, 0, 1);
        //imshow("NOT_KNOWN3", NOT_KNOWN);
	NOT_KNOWN = ~NOT_KNOWN;
        //imshow("NOT_KNOWN4", NOT_KNOWN);

        int flag_up = 0;
        for(int i = 0; i < NOT_KNOWN.rows; ++i) {
            for(int j = 0; j < NOT_KNOWN.cols; ++j) {
                if(NOT_KNOWN.at<uchar>(i, j) <= 100 && NOT_KNOWN.at<uchar>(i, j) >= 80) { flag_up = 1; break; }
                else { NOT_KNOWN.at<uchar>(i, j) = 255; }
            }
            if(flag_up == 1) break;
        }

        int flag_down = 0;
        for(int i = NOT_KNOWN.rows - 1; i >= 0; --i) {
            for(int j = 0; j < NOT_KNOWN.cols; ++j) {
                if(NOT_KNOWN.at<uchar>(i, j) <= 100 && NOT_KNOWN.at<uchar>(i, j) >= 80) { flag_down = 1; break; }
		else { NOT_KNOWN.at<uchar>(i, j) = 255; }
            }
            if(flag_down == 1) break;
        }

        int flag_left = 0;
        for(int j = 0; j < NOT_KNOWN.cols; ++j) {
            for(int i = 0; i < NOT_KNOWN.rows; ++i) {
                if(NOT_KNOWN.at<uchar>(i, j) <= 100 && NOT_KNOWN.at<uchar>(i, j) >= 80) { flag_left = 1; break; }
		else { NOT_KNOWN.at<uchar>(i, j) = 255; }
            }
            if(flag_left == 1) break;
        }

        int flag_right = 0;
        for(int j = NOT_KNOWN.cols - 1; j >= 0; --j) {
            for(int i = 0; i < NOT_KNOWN.rows; ++i) {
                if(NOT_KNOWN.at<uchar>(i, j) <= 100 && NOT_KNOWN.at<uchar>(i, j) >= 80) { flag_right = 1; break; }
		else { NOT_KNOWN.at<uchar>(i, j) = 255; }
            }
            if(flag_right == 1) break;
        }

        cv::Mat zoom3;
        cv::resize(NOT_KNOWN, zoom3, cv::Size(), mul*f, mul*f, cv::INTER_NEAREST);
	cv::medianBlur(zoom3, zoom3, 9);
	cv::medianBlur(zoom3, zoom3, 5);
	cv::medianBlur(zoom3, zoom3, 3);
        //cv::Mat BLAN_known_area3 = cv::Mat::zeros(zoom3.size(), CV_8UC1);
        //cv::medianBlur(zoom3, BLAN_known_area3, 3);
        //cv::blur(zoom3, BLAN_known_area3, cv::Size(5,5));
        //cv::GaussianBlur(zoom3, BLAN_known_area3, cv::Size(3,3), 0, 0);
	//imshow("zoom3", zoom3);

        cv::Mat color_map3 = cv::Mat::zeros(zoom3.size(), CV_8UC3);

        for(int i = 0; i < zoom3.rows; ++i) {
            for(int j = 0; j < zoom3.cols; ++j) {

                if(zoom3.at<uchar>(i, j) < 80) { //墙内
                   color_map3.at<cv::Vec3b>(i, j)[0] = 245;
	           color_map3.at<cv::Vec3b>(i, j)[1] = 81;
	           color_map3.at<cv::Vec3b>(i, j)[2] = 116;
                }
                else if(zoom3.at<uchar>(i, j) > 100) { //墙外
                   color_map3.at<cv::Vec3b>(i, j)[0] = 241;
	           color_map3.at<cv::Vec3b>(i, j)[1] = 119;
	           color_map3.at<cv::Vec3b>(i, j)[2] = 160;
                }
                else { //墙体
                   color_map3.at<cv::Vec3b>(i, j)[0] = 247;
	           color_map3.at<cv::Vec3b>(i, j)[1] = 167;
	           color_map3.at<cv::Vec3b>(i, j)[2] = 205;
                }
            }
        }

	Mat srcI;
	Mat dstI = color_map3.clone();
	threshold(zoom3, srcI, 100, 255, CV_THRESH_BINARY); //二值化
	srcI = ~srcI;
	//imshow("srcI", srcI);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarcy;
	findContours(srcI, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());  //定义外接矩形集合
	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	Point2f rect[4];
	double alpha;
	int boxcenter_x, boxcenter_y;
	for(int i = 0; i < contours.size(); i++)
	{
	    box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
	    alpha = fabs(box[i].angle) > 45 ? 90 - fabs(box[i].angle) : fabs(box[i].angle);
    	    cout << alpha << endl;
	    boxcenter_x = box[i].center.x;
    	    boxcenter_y = box[i].center.y;
	    boundRect[i] = boundingRect(Mat(contours[i]));
	    circle(dstI, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
	    box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
	    //rectangle(dstI, Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
	    for(int j = 0; j < 4; j++)
	    {
		line(dstI, rect[j], rect[(j+1)%4], Scalar(0, 0, 255), 2, 8);  //绘制最小外接矩形每条边
	    }
	}
	imshow("dstI", dstI);
    	//imwrite("to_argb3.bmp", dstI);

	vector<cv::Point2i> Points1;
	vector<cv::Point2i> Points2;
	//输出图像的尺寸与原图一样    
	cv::Size dst_sz(dstI.cols, dstI.rows);
	//指定旋转中心      
	cv::Point2f center(boxcenter_x, boxcenter_y);
	
        for(int i = 0; i < color_map3.rows; ++i) {
            for(int j = 0; j < color_map3.cols; ++j) {

		if(color_map3.at<cv::Vec3b>(i, j)[0] == 247 && color_map3.at<cv::Vec3b>(i, j)[1] == 167 && color_map3.at<cv::Vec3b>(i, j)[2] == 205) {
			Points1.push_back(Point(j, i));
		}

		if(color_map3.at<cv::Vec3b>(i, j)[0] == 245 && color_map3.at<cv::Vec3b>(i, j)[1] == 81 && color_map3.at<cv::Vec3b>(i, j)[2] == 116) {
			Points2.push_back(Point(j, i));
		}
            }
        }
	Mat dstColor(dstI.size(), CV_8UC3, Scalar(241, 119, 160));
	vector<cv::Point2i> dstPoints1 = getRotatePoint(dstColor, Points1, center, alpha);
	for (size_t i = 0; i < dstPoints1.size(); i++)
	{
		cv::circle(dstColor, dstPoints1.at(i), 5, cv::Scalar(247, 167, 205), 2);
	}
	vector<cv::Point2i> dstPoints2 = getRotatePoint(dstColor, Points2, center, alpha);
	for (size_t i = 0; i < dstPoints2.size(); i++)
	{
		cv::circle(dstColor, dstPoints2.at(i), 5, cv::Scalar(245, 81, 116), 2);
	}
	//cv::imshow("Rotation Image", dstColor);

        //cv::blur(dstColor, dstColor, cv::Size(3,3));
	return dstColor;
    }
}


int main()
{
    //例子：5（开区域） 和 map-1（闭区域）
    int data[MAXN], datalen = 0;
    int width = 123, height = 80; // 5

    //获取地图数据
    ifstream file("map_s.txt"); // 5
    while( ! file.eof() )
    file >> data[datalen ++];
    file.close();

    //float x0 = -9.651299476623535; // 5---w = 154, h = 212
    //float y0 = -5.301003932952881; // 5 
    float x0 = -2.328948736190796; // map_s---w = 123, h = 80
    float y0 = -5.480353355407715;
    //float x0 = -1.621787786483765; // map_tw---w = 58, h = 81
    //float y0 = -0.6075371503829956;
    //float x0 = -4.292689800262451; // map_66---w = 183, h = 182
    //float y0 = -7.424565315246582;
    //float x0 = -4.471129894256592; // map_11---w = 159, h = 176
    //float y0 = -5.659699440002441;
    //float x0 = -3.600197076797485; // map_o---w = 210, h = 172
    //float y0 = -9.177140235900879;

    //获得路线数据
    int path_x[MAXN/2], path_y[MAXN/2], len; 
    len = 0;
    ifstream file_x("pathx_s.txt"); // 5x or pathx_o
    while( ! file_x.eof() )
    file_x >> path_x[len ++];
    file_x.close();

    len = 0;
    ifstream file_y("pathy_s.txt"); // 5y or pathy_o
    while( ! file_y.eof() )
    file_y >> path_y[len ++];
    file_y.close();

    cv::Mat asrc = Geo_area(data, width, height, path_x, path_y, len, x0, y0, 1);

    imshow("asrc", asrc);
    //imwrite("to_argb3-.bmp", asrc);

    cvWaitKey(0);

    return 0;
}
