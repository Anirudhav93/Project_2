#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/calib3d/calib3d.hpp"
 
#include <string.h>
#include <iostream>


using namespace cv;
using namespace std;

struct Pos { int x[25], y[25];};
Pos pos;
Mat img = imread("IMG_1.jpg");
static int i = 0;

// mouse callback function
void callback_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {
        if (i >= 20) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        pos.x[i] = x;
        pos.y[i] = y;
        i++;
        rectangle(img, Point(x-5, y-5), Point(x+5, y+5), Scalar(0,255,0), -1);
        imshow("image", img);
    }
}


int main(int argc, char** argv)
{
    // Read image from file 
    //Mat img = imread("IMG_1.jpg");
         
    // if fail to read the image
    if ( img.empty() ) 
    { 
        cout << "Error loading the image" << endl;
        return -1; 
    }

    // Create a window
    namedWindow("image", 1);
    
    // set the callback function for any mouse event
    setMouseCallback("image", callback_Func, NULL);
    
    // show the image
    imshow("image", img);
    
    // Wait until user press some key
    waitKey(0);

    return 0;
    
}
