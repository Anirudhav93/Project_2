#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/calib3d/calib3d.hpp"
 
#include <string.h>
#include <iostream>


using namespace cv;
using namespace std;

//struct Pos { int x[25], y[25];};
//Pos pos1;
//Pos pos2;

vector<Point2f> pos1(25), pos2(25);

Mat img1 = imread("IMG_L.JPG");
Mat img2 = imread("IMG_C.JPG");
static int i = 0, j = 0;

// mouse callback function
void callback1_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {
        if (i >= 20) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        //pos1.x[i] = x;
        //pos1.y[i] = y;
        pos1.push_back(Point(x, y));
        i++;
        namedWindow("image1", WINDOW_NORMAL);
        rectangle(img1, Point(x-50, y-50), Point(x+50, y+50), Scalar(0,255,0), -1);
        imshow("image1", img1);
        waitKey(0);
    }
}

void callback2_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {
        if (j >= 20) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        //pos2.x[j] = x;
        //pos2.y[j] = y;
        pos2.push_back(Point(x, y));
        j++;
        rectangle(img2, Point(x-50, y-50), Point(x+50, y+50), Scalar(0,255,0), -1);
        namedWindow("image2",WINDOW_NORMAL);
        imshow("image2", img2);
        waitKey(0);
    }
}


//template <typename T1, typename T2>
int main(int argc, char** argv)
{
    // Read image from file 
    //Mat img = imread("IMG_1.jpg");
         
    // if fail to read the image
    //if ( img.empty() ) 
    //{ 
    //    cout << "Error loading the image" << endl;
    //    return -1; 
    //}

    // Create a window
    namedWindow("image1", WINDOW_NORMAL);
    namedWindow("image2", WINDOW_NORMAL);

    imshow("image1", img1);
    imshow("image2", img2);


    // set the callback function for any mouse event
    setMouseCallback("image1", callback1_Func, NULL);
    setMouseCallback("image2", callback2_Func, NULL);

    // show the image
    //imshow("image1", img1);
    //imshow("image2", img2);
    
    // Wait until user press some key
    waitKey(0);
    //waitKey(0);


    string filename = "camera.yml";
    FileStorage fs(filename, FileStorage::READ);

    double fx, fy, cx, cy;
    //vector<Point2f> points1, points2;
    //fs["avg_reprojection_error"] >> itNr;
    //itNr = (double) fs["camera_matrix"];
    FileNode n = fs["camera_matrix"];
    FileNode ns = n["data"];
    //itNr = itNr["data"];
    fx = (double) ns[0];
    fy = (double) ns[4];
    cx = (double) ns[2];
    cy = (double) ns[5];
    Point2d pp(cx, cy);

    FileNode d = fs["distortion_coefficients"];
    FileNode ds = d["data"];

    // distortion coefficient vector
    //vector<double> dc(10);
    Mat dc = Mat::eye(5, 1, CV_64F);
    int k = 0;

    for(int p=0;p<5;p++)
        for(int q=0;q<1;q++)
        {
            dc.at<double>(p, q) = (double) ds[k];
            k++;
        }


    Mat E, mask, lr, dt, cam_mat = Mat::eye(3, 3, CV_64F);

    // camera matrix
    k = 0;
    for(int p=0;p<3;p++)
    {
        for(int q=0;q<3;q++)
        {
            cam_mat.at<double>(p, q) = (double) ns[k];
            k++;
        }
    }
    E = findEssentialMat(pos1, pos2, fx, pp, RANSAC, 0.999, 1.0, mask);
    //E = findFundamentalMat(pos1, pos2, CV_FM_RANSAC, 0.999, 1.0, mask);

    dt = getOptimalNewCameraMatrix(cam_mat, dc, img1.size(), 1.0, img1.size());
    
   // vector<Vec4i> epilines1, epilines2;
    
    //computeCorrespondEpilines(pos1, 1, E, epilines1);
    //computeCorrespondEpilines(pos2, 2, E, epilines2);
    //lr = E*lr;

    //for(int m=0;m<E.cols;m++) 
    //{ 
    //    for(int n=0;n<E.rows;n++) 
    //    { 
    //        lr[m][n] = E[m][n]*pos1[n]; 
    //    } 
    //}


    //RNG rng(0);
    //Scalar color(rng(256),rng(256),rng(256));

    //CV_Assert(pos1.size() == pos2.size() &&
    //        pos2.size() == epilines1.size() &&
    //        epilines1.size() == epilines2.size());

    //line(img2,
    //Point(0,-epilines1[i][2]/epilines1[i][1]),
    //Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
    //    color);

    Mat undist_img1 = img1.clone();
    undistort(img1, undist_img1, dt, dc);
    namedWindow("image3", WINDOW_NORMAL);
    imshow("image3", undist_img1);
    //imshow("image2", img2);
    waitKey(0);
    cout << "\nitNr "<< dt << endl;

    return 0;
    
}
