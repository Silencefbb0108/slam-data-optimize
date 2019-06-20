#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>
#include <math.h>

using namespace std;
using namespace cv;

#define MAXN 90000

#define PI 3.1415926535897932384626

cv::Mat houghTransform(cv::Mat & src) {

    cv::Mat res = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    //cvtColor(src, res, COLOR_GRAY2BGR);
    
    //src: 8-bit单通道二值图像
    //lines: 4元素向量(x_1,y_1,x_2,y_2)，(x_1,y_1)和(x_2, y_2)表示线段的两个端点
    //rho = 1(以像素为单位), theta = CV_PI/180(以弧度为单位): 极坐系中r和θ的分辨率
    //threshold: 大于阈值threshold的线段才能够被检測出，threshold值越大，所检测出的直线越少，threshold值越小，所检测出的直线越多
    //minLineLength: 线段的最小长度，小于设定参数的线段被舍弃，显然该值越大，所检测出的直线越少，该值越小，所检测出的直线越多
    //maxLineGap: 同一直线上的点与点之间被连接的最大间隙
    vector<Vec4i> lines;
    HoughLinesP(src, lines, 1, CV_PI/180, 16, 8, 6);
    
    for( size_t i = 0; i < lines.size(); i++ ){
        cv::line(res, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255,255,255), 2, 8);
        //cout << lines[i][1] << "," << lines[i][0] << " " << lines[i][3] << "," << lines[i][2] << endl; //0-1-2-3: col-row-col-row
    }
    
    for(size_t i = 0; i < lines.size()-1; i++){
        for(size_t j = i+1; j < lines.size(); j++){
            if((abs(lines[i][1]-lines[j][1]) <= 3 && abs(lines[i][0]-lines[j][0]) < 7) || (abs(lines[i][1]-lines[j][1]) < 7 && abs(lines[i][0]-lines[j][0]) <= 3)){
               cv::line(res, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[j][0], lines[j][1]), cv::Scalar(255,255,255), 2, 8);
               //cout << lines[i][1] << "," << lines[i][0] << " " << lines[j][1] << "," << lines[j][0] << endl;
            }
            else if((abs(lines[i][1]-lines[j][3]) <= 3 && abs(lines[i][0]-lines[j][2]) < 7) || (abs(lines[i][1]-lines[j][3]) < 7 && abs(lines[i][0]-lines[j][2]) <= 3)){
               cv::line(res, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[j][2], lines[j][3]), cv::Scalar(255,255,255), 2, 8);
               //cout << lines[i][1] << "," << lines[i][0] << " " << lines[j][3] << "," << lines[j][2] << endl;
            }
            else if((abs(lines[i][3]-lines[j][1]) <= 3 && abs(lines[i][2]-lines[j][0]) < 7) || (abs(lines[i][3]-lines[j][1]) < 7 && abs(lines[i][2]-lines[j][0]) <= 3)){
               cv::line(res, cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][0], lines[j][1]), cv::Scalar(255,255,255), 2, 8);
               //cout << lines[i][3] << "," << lines[i][2] << " " << lines[j][1] << "," << lines[j][0] << endl;
            }
            else if((abs(lines[i][3]-lines[j][3]) <= 3 && abs(lines[i][2]-lines[j][2]) < 7) || (abs(lines[i][3]-lines[j][3]) < 7 && abs(lines[i][2]-lines[j][2]) <= 3)){
               cv::line(res, cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][2], lines[j][3]), cv::Scalar(255,255,255), 2, 8);
               //cout << lines[i][3] << "," << lines[i][2] << " " << lines[j][3] << "," << lines[j][2] << endl;
            }
        }
    }

    cv::Mat res2 = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    cvtColor(res, res2, COLOR_BGR2GRAY);

    return res2;
}

/*
* @brief 对输入图像进行细化, 骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1, 1代表有元素，0代表为空白
* @param maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
* @return 为对src细化后的输出图像, 格式与src格式相同，元素中只有0与1, 1代表有元素，0代表为空白
*/

cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1) {

    assert(src.type() == CV_8UC1);
    cv::Mat dst;
    int width = src.cols;
    int height = src.rows;
    src.copyTo(dst);
    int count = 0;  //记录迭代次数  
    while (true)
    {
	count++;
	if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达
	    break;
	std::vector<uchar *> mFlag; //用于标记需要删除的点  
	//对点标记  
	for (int i = 0; i < height; ++i)
	{
	     uchar * p = dst.ptr<uchar>(i);
	     for (int j = 0; j < width; ++j)
	     {
		 //如果满足四个条件，进行标记  
		 //  p9 p2 p3  
		 //  p8 p1 p4  
		 //  p7 p6 p5  
		 uchar p1 = p[j];
		 if (p1 != 1) continue;
		 uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
		 uchar p8 = (j == 0) ? 0 : *(p + j - 1);
		 uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
		 uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
		 uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
		 uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
		 uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
		 uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
		 if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
		 {
		     int ap = 0;
		     if (p2 == 0 && p3 == 1) ++ap;
		     if (p3 == 0 && p4 == 1) ++ap;
		     if (p4 == 0 && p5 == 1) ++ap;
		     if (p5 == 0 && p6 == 1) ++ap;
		     if (p6 == 0 && p7 == 1) ++ap;
		     if (p7 == 0 && p8 == 1) ++ap;
		     if (p8 == 0 && p9 == 1) ++ap;
		     if (p9 == 0 && p2 == 1) ++ap;
 
		     if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
		     {
			 //标记  
			 mFlag.push_back(p + j);
		     }
		 }
	     }
        }
 
	//将标记的点删除  
	for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
	{
	     **i = 0;
	}
 
	//直到没有点满足，算法结束  
	if (mFlag.empty())
	{
	    break;
	}
	else
	{
	    mFlag.clear();//将mFlag清空  
	}
 
	//对点标记  
	for (int i = 0; i < height; ++i)
	{
	     uchar * p = dst.ptr<uchar>(i);
	     for (int j = 0; j < width; ++j)
	     {
		  //如果满足四个条件，进行标记  
		  //  p9 p2 p3  
		  //  p8 p1 p4  
		  //  p7 p6 p5  
		  uchar p1 = p[j];
		  if (p1 != 1) continue;
		  uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
		  uchar p8 = (j == 0) ? 0 : *(p + j - 1);
		  uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
	          uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
		  uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
		  uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
		  uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
		  uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
 
		  if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
		  {
		      int ap = 0;
		      if (p2 == 0 && p3 == 1) ++ap;
		      if (p3 == 0 && p4 == 1) ++ap;
		      if (p4 == 0 && p5 == 1) ++ap;
		      if (p5 == 0 && p6 == 1) ++ap;
		      if (p6 == 0 && p7 == 1) ++ap;
		      if (p7 == 0 && p8 == 1) ++ap;
		      if (p8 == 0 && p9 == 1) ++ap;
		      if (p9 == 0 && p2 == 1) ++ap;
 
		      if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
		      {
			  //标记
			  mFlag.push_back(p + j);
		      }
		  }
	     }
	}
 
	//将标记的点删除  
	for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
	{
	     **i = 0;
	}
 
	//直到没有点满足，算法结束  
	if (mFlag.empty())
	{
	    break;
	}
	else
	{
	    mFlag.clear();//将mFlag清空  
	}
    }
    dst = dst * 255;

    return dst;
}

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

/*腐蚀和膨胀是对白色部分（高亮部分）而言，不是黑色部分。
  膨胀是对图像高亮部分进行膨胀，效果图拥有比原图更大的高亮区域。
  腐蚀是对图像高亮部分进行腐蚀，效果图拥有比原图更小的高亮区域。*/

cv::Mat Geo_area(int* data, int width, int height, int* path_x, int* path_y, int len, float x0, float y0, float f) {

    cv::Mat src = cv::Mat::zeros(height, width, CV_8UC1);
    for(int i = 0; i < src.rows; ++i) {
        for(int j = 0; j < src.cols; ++j) {
            src.at<uchar>(i, j) = data[width*i+j];
        }
    }
    
    cv::Mat src2 = cv::Mat::zeros(src.size(), CV_8UC1);
    src2 = (80 <= src & src <= 100);

    if(countNonZero(src2) == 0) {

       cv::Mat zoom;
       cv::resize(src, zoom, cv::Size(), 4*f, 4*f, cv::INTER_NEAREST);
       cv::Mat color_map = cv::Mat::zeros(zoom.size(), CV_8UC3);

       for(int i = 0; i < zoom.rows; ++i) {
           for(int j = 0; j < zoom.cols; ++j) {
               if(zoom.at<uchar>(i, j) < 80) { //墙内
                  color_map.at<cv::Vec3b>(i, j)[0] = 143;
	          color_map.at<cv::Vec3b>(i, j)[1] = 255;
	          color_map.at<cv::Vec3b>(i, j)[2] = 206;
               }
               else if(zoom.at<uchar>(i, j) > 100) { //墙外
                  color_map.at<cv::Vec3b>(i, j)[0] = 47;
	          color_map.at<cv::Vec3b>(i, j)[1] = 65;
	          color_map.at<cv::Vec3b>(i, j)[2] = 145;
               }
               else { //墙体
                  color_map.at<cv::Vec3b>(i, j)[0] = 116;
	          color_map.at<cv::Vec3b>(i, j)[1] = 255;
	          color_map.at<cv::Vec3b>(i, j)[2] = 195;
               }
           }   
       }

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

	if(y_pix_col2[h5] < 0) y_pix_col2[h5] = 0;
	else if(y_pix_col2[h5] > src.rows-1) y_pix_col2[h5] = src.rows-1;

	if(x_pix_row2[h5] < 0) x_pix_row2[h5] = 0;
	else if(x_pix_row2[h5] > src.cols-1) x_pix_row2[h5] = src.cols-1;

	if(y_pix_col2[h6] < 0) y_pix_col2[h6] = 0;
	else if(y_pix_col2[h6] > src.rows-1) y_pix_col2[h6] = src.rows-1;

	if(x_pix_row2[h6] < 0) x_pix_row2[h6] = 0;
	else if(x_pix_row2[h6] > src.cols-1) x_pix_row2[h6] = src.cols-1;

	if(y_pix_col2[h7] < 0) y_pix_col2[h7] = 0;
	else if(y_pix_col2[h7] > src.rows-1) y_pix_col2[h7] = src.rows-1;

	if(x_pix_row2[h7] < 0) x_pix_row2[h7] = 0;
	else if(x_pix_row2[h7] > src.cols-1) x_pix_row2[h7] = src.cols-1;

	if(y_pix_col2[h8] < 0) y_pix_col2[h8] = 0;
	else if(y_pix_col2[h8] > src.rows-1) y_pix_col2[h8] = src.rows-1;

	if(x_pix_row2[h8] < 0) x_pix_row2[h8] = 0;
	else if(x_pix_row2[h8] > src.cols-1) x_pix_row2[h8] = src.cols-1;

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
	int nccomps3 = cv::connectedComponentsWithStats(B_known_area2, labels3, stats3, centroids3, 4); //连通域分析，B_known_area2为8位单通道二值图像

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
	cv::Mat element2 = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        erode(BLA_known_area2, BLA_known_area2, element2);
	dilate(BLA_known_area2, BLA_known_area2, element2);
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


        cv::Mat BLANK2 = cv::Mat::zeros(src.size(), CV_8UC1);
        BLANK2 = (BLAN_known_area2 < 80);
        cv::Mat BLANK3 = cv::Mat::zeros(src.size(), CV_8UC1);
        BLANK2.copyTo(BLANK3);

        for(int i = 1; i < src.rows-1; ++i) {
            for(int j = 1; j < src.cols-1; ++j) {
		if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j-1) == 0 && BLANK3.at<uchar>(i-1, j) == 0) {
                   BLANK2.at<uchar>(i-1, j-1) = 255;
                }
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j+1) == 0 && BLANK3.at<uchar>(i+1, j) == 0) {
                   BLANK2.at<uchar>(i+1, j+1) = 255;
                }
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i, j-1) == 0 && BLANK3.at<uchar>(i+1, j) == 0) {
                   BLANK2.at<uchar>(i+1, j-1) = 255;
                }
                if(BLANK3.at<uchar>(i, j) == 255 && BLANK3.at<uchar>(i-1, j) == 0 && BLANK3.at<uchar>(i, j+1) == 0) {
                   BLANK2.at<uchar>(i-1, j+1) = 255;
                }
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
            }
        }
        //imshow("BLANK3", BLANK3);

        cv::Mat BARRIER2 = cv::Mat::zeros(src.size(), CV_8UC1);
        BARRIER2 = (80 <= BLAN_known_area2 & BLAN_known_area2 <= 100);
        cv::Mat BARRIER3 = cv::Mat::zeros(src.size(), CV_8UC1);
        bitwise_and(BLANK2, BARRIER2, BARRIER3);
        //imshow("BARRIER3", BARRIER3);
        cv::Mat NOT_KNOWN = cv::Mat::zeros(src.size(), CV_8UC1);
        bitwise_or(BLANK3, BARRIER3, NOT_KNOWN);
        //imshow("NOT_KNOWN", NOT_KNOWN);
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
        cv::resize(NOT_KNOWN, zoom3, cv::Size(), 4*f, 4*f, cv::INTER_NEAREST);

        cv::Mat color_map3 = cv::Mat::zeros(zoom3.size(), CV_8UC3);

        for(int i = 0; i < zoom3.rows; ++i) {
            for(int j = 0; j < zoom3.cols; ++j) {

                if(zoom3.at<uchar>(i, j) < 80) { //墙内
                   color_map3.at<cv::Vec3b>(i, j)[0] = 143;
	           color_map3.at<cv::Vec3b>(i, j)[1] = 255;
	           color_map3.at<cv::Vec3b>(i, j)[2] = 206;
                }
                else if(zoom3.at<uchar>(i, j) > 100) { //墙外
                   color_map3.at<cv::Vec3b>(i, j)[0] = 47;
	           color_map3.at<cv::Vec3b>(i, j)[1] = 65;
	           color_map3.at<cv::Vec3b>(i, j)[2] = 145;
                }
                else { //墙体
                   color_map3.at<cv::Vec3b>(i, j)[0] = 116;
	           color_map3.at<cv::Vec3b>(i, j)[1] = 255;
	           color_map3.at<cv::Vec3b>(i, j)[2] = 195;
                }
            }   
        }

	return color_map3;
    }
}

double angle(int x1, int y1, int x2, int y2, int x3, int y3) {
    double aa = pow(x1 - x2, 2) + pow(y1 - y2, 2);
    double bb = pow(x3 - x2, 2) + pow(y3 - y2, 2);
    double cc = pow(x3 - x1, 2) + pow(y3 - y1, 2);
    double C = acos((aa + bb - cc)/(2 * sqrt(aa) * sqrt(bb)));

    return C;  
}

void curve_out(int* x, int* y, int len, int l, int C) {

    for(int i = 0; i < len-l; i +=l) {
        if(angle(x[i], y[i], x[i+l/2], y[i+l/2], x[i+l], y[i+l]) > PI/180*C) {
           for(int j = i+1; j < i+l; j ++) {
               x[j] = (x[j-1] + x[j] + x[j+1])/3;
               y[j] = (y[j-1] + y[j] + y[j+1])/3;
           }
        }
    }
}

void linear_fit0(int* x_pre, int* y_pre, int* x, int* y, int len) {

    if(len < 3) {
       for(int i = 0; i < len; ++i) {
	   x[i] = x_pre[i];
	   y[i] = y_pre[i];
       }
    }

    else {
       x[0] = x_pre[0]; y[0] = y_pre[0];
       x[1] = x_pre[1]; y[1] = y_pre[1];
       x[len-2] = x_pre[len-2]; y[len-2] = y_pre[len-2];
       x[len-1] = x_pre[len-1]; y[len-1] = y_pre[len-1];

       for(int i = 2; i < len-2; ++i) {

           if((x_pre[i-1] != x_pre[i] && x_pre[i-1] == x_pre[i+1] && (x_pre[i-2] == x_pre[i-1] || x_pre[i+2] == x_pre[i+1])) || 
                   (y_pre[i-1] != y_pre[i] && y_pre[i-1] == y_pre[i+1] && (y_pre[i-2] == y_pre[i-1] || y_pre[i+2] == y_pre[i+1]))) {
              x[i] = (x_pre[i-1] + x_pre[i+1])/2; 
              y[i] = (y_pre[i-1] + y_pre[i+1])/2;
           }

           else {
              x[i] = x_pre[i]; y[i] = y_pre[i];
           }
       }
    }

    for(int i = 0; i < len; i ++) {
        x_pre[i] = x[i]; y_pre[i] = y[i];
    }
}

void linear_fit1(int* x_pre, int* y_pre, int* x, int* y, int len) {

    if(len < 3) {
       for(int i = 0; i < len; ++i) {
	   x[i] = x_pre[i];
	   y[i] = y_pre[i];
       }
    }

    else {
       x[0] = x_pre[0]; y[0] = y_pre[0];
       x[1] = x_pre[1]; y[1] = y_pre[1];
       x[len-2] = x_pre[len-2]; y[len-2] = y_pre[len-2];
       x[len-1] = x_pre[len-1]; y[len-1] = y_pre[len-1];

       for(int i = 2; i < len-2; ++i) {

           if(x_pre[i-2] == x_pre[i-1] && x_pre[i-1] == x_pre[i] && (x_pre[i+1] > x_pre[i] && x_pre[i+1] < x_pre[i+2]) && (y_pre[i+1] >= y_pre[i] && y_pre[i+1] >= y_pre[i+2])) {
              x[i] = x_pre[i]; 
              y[i] = y_pre[i+1];
           }

           else if(x_pre[i+2] == x_pre[i+1] && x_pre[i+1] == x_pre[i] && (x_pre[i-1] < x_pre[i] && x_pre[i-1] > x_pre[i-2]) && (y_pre[i-1] >= y_pre[i] && y_pre[i-1] >= y_pre[i-2])) {
              x[i] = x_pre[i]; 
              y[i] = y_pre[i-1];
           }

           else if(x_pre[i-2] == x_pre[i-1] && x_pre[i-1] == x_pre[i] && (x_pre[i+1] > x_pre[i] && x_pre[i+1] < x_pre[i+2]) && (y_pre[i+1] <= y_pre[i] && y_pre[i+1] <= y_pre[i+2])) {
              x[i] = x_pre[i]; 
              y[i] = y_pre[i+1];
           }

           else if(x_pre[i+2] == x_pre[i+1] && x_pre[i+1] == x_pre[i] && (x_pre[i-1] < x_pre[i] && x_pre[i-1] > x_pre[i-2]) && (y_pre[i-1] <= y_pre[i] && y_pre[i-1] <= y_pre[i-2])) {
              x[i] = x_pre[i]; 
              y[i] = y_pre[i-1];
           }

           else if(y_pre[i-2] == y_pre[i-1] && y_pre[i-1] == y_pre[i] && (y_pre[i+1] > y_pre[i] && y_pre[i+1] < y_pre[i+2]) && (x_pre[i+1] >= x_pre[i] && x_pre[i+1] >= x_pre[i+2])) {
              x[i] = x_pre[i+1]; 
              y[i] = y_pre[i];
           }

           else if(y_pre[i+2] == y_pre[i+1] && y_pre[i+1] == y_pre[i] && (y_pre[i-1] < y_pre[i] && y_pre[i-1] > y_pre[i-2]) && (x_pre[i-1] >= x_pre[i] && x_pre[i-1] >= x_pre[i-2])) {
              x[i] = x_pre[i-1]; 
              y[i] = y_pre[i];
           }

           else if(y_pre[i-2] == y_pre[i-1] && y_pre[i-1] == y_pre[i] && (y_pre[i+1] > y_pre[i] && y_pre[i+1] < y_pre[i+2]) && (x_pre[i+1] <= x_pre[i] && x_pre[i+1] <= x_pre[i+2])) {
              x[i] = x_pre[i+1]; 
              y[i] = y_pre[i];
           }

           else if(y_pre[i+2] == y_pre[i+1] && y_pre[i+1] == y_pre[i] && (y_pre[i-1] < y_pre[i] && y_pre[i-1] > y_pre[i-2]) && (x_pre[i-1] <= x_pre[i] && x_pre[i-1] <= x_pre[i-2])) {
              x[i] = x_pre[i-1]; 
              y[i] = y_pre[i];
           }

           else {
              x[i] = x_pre[i]; y[i] = y_pre[i];
           }

       }
    }

    for(int i = 0; i < len; i ++) {
        x_pre[i] = x[i]; y_pre[i] = y[i];
    }
}

void linear_fit(int* x_pre, int* y_pre, int* x, int* y, int len) {

    if(len < 3) {
       for(int i = 0; i < len; ++i) {
	   x[i] = x_pre[i];
	   y[i] = y_pre[i];
       }
    }
    else {
       x[0] = x_pre[0]; y[0] = y_pre[0];
       x[1] = (x_pre[0] + x_pre[1] + x_pre[2])/3; 
       y[1] = (y_pre[0] + y_pre[1] + y_pre[2])/3;
       x[len-2] = (x_pre[len-3] + x_pre[len-2] + x_pre[len-1])/3; 
       y[len-2] = (y_pre[len-3] + y_pre[len-2] + y_pre[len-1])/3;
       x[len-1] = x_pre[len-1]; y[len-1] = y_pre[len-1];

       for(int i = 2; i < len-2; ++i) {

           x[i] = (x_pre[i-1] + x_pre[i] + x_pre[i+1])/3;
	   y[i] = (y_pre[i-1] + y_pre[i] + y_pre[i+1])/3;

       }
    }

    curve_out(x, y, len, 4, 175);

    int k, ie = 1, je = 1, istart = 1, iend = 1;
    while(ie < len-1) {
        if(angle(x[ie-1], y[ie-1], x[ie], y[ie], x[ie+1], y[ie+1]) > PI/180*175) {

           je = ie+1;
           if(je < len-1) {
              istart = ie;
              k = 0;
              while(je < len-1 && angle(x[je-1], y[je-1], x[je], y[je], x[je+1], y[je+1]) > PI/180*175) {
                  k ++; je ++;
              }
              iend = je;
              for(int cc = istart+1; cc <= istart+k; cc ++) {
                  x[cc] = (x[cc-1] + x[cc] + x[cc+1])/3;
                  y[cc] = (y[cc-1] + y[cc] + y[cc+1])/3;
              }
              ie = iend;
           }
           else {break;}
        }
        else {ie ++;}
    }

}

void linear_fit2(int* x_pre, int* y_pre, int* x, int* y, int len, int n) {
    
    for(int i = 1; i <= n; i ++) {
        linear_fit(x_pre, y_pre, x, y, len);

        for(int i = 0; i < len; i ++) {
            x_pre[i] = x[i]; y_pre[i] = y[i];
        }
    }
}

cv::Mat Pathprocess(cv::Mat & src, int* path_x, int* path_y, int len, float x0, float y0, float f) {

    int x_row[len], y_col[len];
    int x_pix_row[len], y_pix_col[len];

    memset(x_row, 0, sizeof(x_row));
    memset(y_col, 0, sizeof(y_col));
    memset(x_pix_row, 0, sizeof(x_pix_row));
    memset(y_pix_col, 0, sizeof(y_pix_col));

    for(int i = 0; i < len; ++i) {
        x_row[i] = int(path_x[i]/5.0 - x0*20);
        y_col[i] = int(path_y[i]/5.0 - y0*20);
        x_pix_row[i] = (4*f)*(src.cols/(4*f) - y_col[i]);
        y_pix_col[i] = (4*f)*(src.rows/(4*f) - x_row[i]);

    }
    
    int x[len], y[len];
    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));
    linear_fit0(x_pix_row, y_pix_col, x, y, len);
    linear_fit1(x_pix_row, y_pix_col, x, y, len);
    linear_fit2(x_pix_row, y_pix_col, x, y, len, 2);

    cv::Mat src2 = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    cvtColor(src, src2, cv::COLOR_BGR2GRAY);

    cv::Mat bac = cv::Mat::zeros(1, 1, CV_8UC3);
    bac.at<cv::Vec3b>(0, 0)[0] = 47;
    bac.at<cv::Vec3b>(0, 0)[1] = 65;
    bac.at<cv::Vec3b>(0, 0)[2] = 145;
    cv::Mat bac_gray = cv::Mat::zeros(1, 1, CV_8UC1);
    cvtColor(bac, bac_gray, cv::COLOR_BGR2GRAY);

    cv::Mat bar = cv::Mat::zeros(1, 1, CV_8UC3);
    bar.at<cv::Vec3b>(0, 0)[0] = 143;
    bar.at<cv::Vec3b>(0, 0)[1] = 255;
    bar.at<cv::Vec3b>(0, 0)[2] = 206;
    cv::Mat bar_gray = cv::Mat::zeros(1, 1, CV_8UC1);
    cvtColor(bar, bar_gray, cv::COLOR_BGR2GRAY);

    cv::Mat B_src2 = (src2 == bar_gray.at<uchar>(0, 0));
    if(countNonZero(B_src2) != 0 && len > 9) {

       int ie = 5, je = 6, istart = 5, iend = 6;
       while(ie < len-4) {
	   if(angle(x[ie-1], y[ie-1], x[ie], y[ie], x[ie+1], y[ie+1]) > PI/180*172) {

	      je = ie+1;
	      if(je < len-4) {
	         istart = ie;
	         while(angle(x[je-1], y[je-1], x[je], y[je], x[je+1], y[je+1]) > PI/180*172 && je < len-4) {
	             je ++;
	         }
	         iend = je;
	         cv::line(src, cv::Point(x[istart], y[istart]), cv::Point(x[iend], y[iend]), cv::Scalar(255,255,255), 2, CV_AA);

	         ie = iend;
	      }
	      else {
                 cv::line(src, cv::Point(x[ie], y[ie]), cv::Point(x[je], y[je]), cv::Scalar(255,255,255), 2, CV_AA);
                 break;
              }
	   }
	   else {
              ie ++;
              cv::line(src, cv::Point(x[ie-1], y[ie-1]), cv::Point(x[ie], y[ie]), cv::Scalar(255,255,255), 2, CV_AA);
           }
       }

    }

    int up_src2 = 0, down_src2 = src2.rows-1, left_src2 = 0, right_src2 = src2.cols-1;
    int flag_up = 0;
    for(int i = 0; i < src2.rows; ++i) {
        for(int j = 0; j < src2.cols; ++j) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { up_src2 = i; flag_up = 1; break; }
        }
        if(flag_up == 1) break;
    }

    int flag_down = 0;
    for(int i = src2.rows - 1; i >= 0; --i) {
        for(int j = 0; j < src2.cols; ++j) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { down_src2 = i; flag_down = 1; break; }
        }
        if(flag_down == 1) break;
    }

    int flag_left = 0;
    for(int j = 0; j < src2.cols; ++j) {
        for(int i = 0; i < src2.rows; ++i) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { left_src2 = j; flag_left = 1; break; }
        }
        if(flag_left == 1) break;
    }

    int flag_right = 0;
    for(int j = src2.cols - 1; j >= 0; --j) {
        for(int i = 0; i < src2.rows; ++i) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { right_src2 = j; flag_right = 1; break; }
        }
        if(flag_right == 1) break;
    }

    cv::Mat src3 = cv::Mat::zeros(down_src2-up_src2+1, right_src2-left_src2+1, CV_8UC3);
    src3 = src(cv::Range(up_src2, down_src2+1), cv::Range(left_src2, right_src2+1));

    //添加最后一行、最后一列
    cv::Mat last_row(10, right_src2-left_src2+1, CV_8UC3, cv::Scalar(47, 65, 145));
    src3.push_back(last_row); //行: down_src2-up_src2+6，列: right_src2-left_src2+1
    src3 = src3.t(); //行: right_src2-left_src2+1，列: down_src2-up_src2+6
    cv::Mat last_col(10, down_src2-up_src2+11, CV_8UC3, cv::Scalar(47, 65, 145));
    src3.push_back(last_col); //行: right_src2-left_src2+6，列: down_src2-up_src2+6
    src3 = src3.t(); //行: down_src2-up_src2+6，列: right_src2-left_src2+6
    flip(src3, src3, -1);

    //添加第一行、第一列
    cv::Mat first_row(10, right_src2-left_src2+11, CV_8UC3, cv::Scalar(47, 65, 145));
    src3.push_back(first_row); //行: down_src2-up_src2+11，列: right_src2-left_src2+6
    src3 = src3.t(); //行: right_src2-left_src2+6，列: down_src2-up_src2+11
    cv::Mat first_col(10, down_src2-up_src2+21, CV_8UC3, cv::Scalar(47, 65, 145));
    src3.push_back(first_col); //行: right_src2-left_src2+11，列: down_src2-up_src2+11
    src3 = src3.t(); //行: down_src2-up_src2+11，列: right_src2-left_src2+11
    flip(src3, src3, -1);

    return src3;
}

struct Coor_point{
     int x1, y1, x2, y2;
};

Coor_point Endpoint(cv::Mat & src, int* path_x, int* path_y, int len, float x0, float y0, float f) {

    int x_row[len], y_col[len];
    int x_pix_row[len], y_pix_col[len];

    memset(x_row, 0, sizeof(x_row));
    memset(y_col, 0, sizeof(y_col));
    memset(x_pix_row, 0, sizeof(x_pix_row));
    memset(y_pix_col, 0, sizeof(y_pix_col));

    for(int i = 0; i < len; ++i) {
        x_row[i] = int(path_x[i]/5.0 - x0*20);
        y_col[i] = int(path_y[i]/5.0 - y0*20);
        x_pix_row[i] = (4*f)*(src.cols/(4*f) - y_col[i]);
        y_pix_col[i] = (4*f)*(src.rows/(4*f) - x_row[i]);
    }
    
    int x[len], y[len];
    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));
    linear_fit0(x_pix_row, y_pix_col, x, y, len);
    linear_fit1(x_pix_row, y_pix_col, x, y, len);
    linear_fit2(x_pix_row, y_pix_col, x, y, len, 2);

    cv::Mat src2 = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
    cvtColor(src, src2, cv::COLOR_BGR2GRAY);

    cv::Mat bac = cv::Mat::zeros(1, 1, CV_8UC3);
    bac.at<cv::Vec3b>(0, 0)[0] = 47;
    bac.at<cv::Vec3b>(0, 0)[1] = 65;
    bac.at<cv::Vec3b>(0, 0)[2] = 145;
    cv::Mat bac_gray = cv::Mat::zeros(1, 1, CV_8UC1);
    cvtColor(bac, bac_gray, cv::COLOR_BGR2GRAY);

    cv::Mat bar = cv::Mat::zeros(1, 1, CV_8UC3);
    bar.at<cv::Vec3b>(0, 0)[0] = 143;
    bar.at<cv::Vec3b>(0, 0)[1] = 255;
    bar.at<cv::Vec3b>(0, 0)[2] = 206;
    cv::Mat bar_gray = cv::Mat::zeros(1, 1, CV_8UC1);
    cvtColor(bar, bar_gray, cv::COLOR_BGR2GRAY);

    int up_src2 = 0, down_src2 = src2.rows-1, left_src2 = 0, right_src2 = src2.cols-1;
    int flag_up = 0;
    for(int i = 0; i < src2.rows; ++i) {
        for(int j = 0; j < src2.cols; ++j) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { up_src2 = i; flag_up = 1; break; }
        }
        if(flag_up == 1) break;
    }

    int flag_down = 0;
    for(int i = src2.rows - 1; i >= 0; --i) {
        for(int j = 0; j < src2.cols; ++j) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { down_src2 = i; flag_down = 1; break; }
        }
        if(flag_down == 1) break;
    }

    int flag_left = 0;
    for(int j = 0; j < src2.cols; ++j) {
        for(int i = 0; i < src2.rows; ++i) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { left_src2 = j; flag_left = 1; break; }
        }
        if(flag_left == 1) break;
    }

    int flag_right = 0;
    for(int j = src2.cols - 1; j >= 0; --j) {
        for(int i = 0; i < src2.rows; ++i) {
            if(src2.at<uchar>(i, j) != bac_gray.at<uchar>(0, 0)) { right_src2 = j; flag_right = 1; break; }
        }
        if(flag_right == 1) break;
    }

    struct Coor_point points;
    cv::Mat B_src2 = (src2 == bar_gray.at<uchar>(0, 0));
    if(countNonZero(B_src2) == 0 || len <= 9) {
       points.x1 = 0;
       points.y1 = 0;
       points.x2 = 0;
       points.y2 = 0;
       return points;
    }

    else {

       points.x1 = x[5]-(left_src2-1)+10;
       points.y1 = y[5]-(up_src2-1)+10;
       points.x2 = x[len-4]-(left_src2-1)+10;
       points.y2 = y[len-4]-(up_src2-1)+10;
    }
    
    return points;
}

Coor_point Geo(int* data, int width, int height, int* path_x, int* path_y, int len, float x0, float y0, float f, cv::Mat & bsrc)
{

    cv::Mat zoom = Geo_area(data, width, height, path_x, path_y, len, x0, y0, f);
    cv::Mat src3 = Pathprocess(zoom, path_x, path_y, len, x0, y0, f);

    cv::Mat asrc = cv::Mat::zeros(src3.rows, src3.cols, CV_8UC4);
    for(int i = 0; i < asrc.rows; ++i) {
        for(int j = 0; j < asrc.cols; ++j) {

            if(src3.at<cv::Vec3b>(i, j)[0] == 47 && src3.at<cv::Vec3b>(i, j)[1] == 65 && src3.at<cv::Vec3b>(i, j)[2] == 145) {

               asrc.at<cv::Vec4b>(i, j)[0] = src3.at<cv::Vec3b>(i, j)[0];
	       asrc.at<cv::Vec4b>(i, j)[1] = src3.at<cv::Vec3b>(i, j)[1];
	       asrc.at<cv::Vec4b>(i, j)[2] = src3.at<cv::Vec3b>(i, j)[2];
               asrc.at<cv::Vec4b>(i, j)[3] = 0x00;
            }

	    else if(src3.at<cv::Vec3b>(i, j)[0] == 143 && src3.at<cv::Vec3b>(i, j)[1] == 255 && src3.at<cv::Vec3b>(i, j)[2] == 206) {

               asrc.at<cv::Vec4b>(i, j)[0] = src3.at<cv::Vec3b>(i, j)[0];
	       asrc.at<cv::Vec4b>(i, j)[1] = src3.at<cv::Vec3b>(i, j)[1];
	       asrc.at<cv::Vec4b>(i, j)[2] = src3.at<cv::Vec3b>(i, j)[2];
               asrc.at<cv::Vec4b>(i, j)[3] = 0x5a;
            }

            else if(src3.at<cv::Vec3b>(i, j)[0] == 116 && src3.at<cv::Vec3b>(i, j)[1] == 255 && src3.at<cv::Vec3b>(i, j)[2] == 195) {

               asrc.at<cv::Vec4b>(i, j)[0] = src3.at<cv::Vec3b>(i, j)[0];
	       asrc.at<cv::Vec4b>(i, j)[1] = src3.at<cv::Vec3b>(i, j)[1];
	       asrc.at<cv::Vec4b>(i, j)[2] = src3.at<cv::Vec3b>(i, j)[2];
               asrc.at<cv::Vec4b>(i, j)[3] = 0xff;
            }

	    else {

               asrc.at<cv::Vec4b>(i, j)[0] = 181;
	       asrc.at<cv::Vec4b>(i, j)[1] = 233;
	       asrc.at<cv::Vec4b>(i, j)[2] = 215;
               asrc.at<cv::Vec4b>(i, j)[3] = 0xff;
            }
        }
    }

    bsrc = asrc;

    Coor_point points = Endpoint(zoom, path_x, path_y, len, x0, y0, f);
    return points;
}

int main()
{
    //例子：5（开区域） 和 map-1（闭区域）
    int data[MAXN], datalen = 0;
    int width = 154, height = 212; // 5
    //int width = 210, height = 172; // map-1

    //获取地图数据
    ifstream file("5.txt"); // 5 or map-1
    while( ! file.eof() )
    file >> data[datalen ++];
    file.close();

    float x0 = -9.651299476623535; // 5
    float y0 = -5.301003932952881; // 5 
    //float x0 = -3.600197076797485; // map-1
    //float y0 = -9.177140235900879; // map-1

    //获得路线数据
    int path_x[MAXN/2], path_y[MAXN/2], len; 
    len = 0;
    ifstream file_x("5x.txt"); // 5x or pathx_o
    while( ! file_x.eof() )
    file_x >> path_x[len ++];
    file_x.close();

    len = 0;
    ifstream file_y("5y.txt"); // 5y or pathy_o
    while( ! file_y.eof() )
    file_y >> path_y[len ++];
    file_y.close();

    cv::Mat bsrc;

    Coor_point points = Geo(data, width, height, path_x, path_y, len, x0, y0, 1, bsrc);

    imshow("bsrc", bsrc);
    imwrite("to_argb.png", bsrc);

    cvWaitKey(0);

    return 0;
}

