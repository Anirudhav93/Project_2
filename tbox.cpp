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

Mat img1 = imread("IMG_1.jpg");
Mat img2 = imread("IMG_2.jpg");
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
        rectangle(img1, Point(x-5, y-5), Point(x+5, y+5), Scalar(0,255,0), -1);
        imshow("image1", img1);
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
        rectangle(img2, Point(x-5, y-5), Point(x+5, y+5), Scalar(0,255,0), -1);
        imshow("image2", img2);
    }
}


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
    namedWindow("image1", 1);
    namedWindow("image2", 1);

    // set the callback function for any mouse event
    setMouseCallback("image1", callback1_Func, NULL);
    setMouseCallback("image2", callback2_Func, NULL);

    // show the image
    imshow("image1", img1);
    imshow("image2", img2);
    
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

    Mat E, mask;
    E = findEssentialMat(pos1, pos2, fx, pp, RANSAC, 0.999, 1.0, mask);

    cout << "\nitNr "<< E << endl;

    return 0;
    
}
