#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "~/OpenCV/opencv-2.4.13/modules/nonfree/include/opencv2/nonfree/features2d.hpp"
#include <string.h>
#include <iostream>
using namespace cv;
using namespace std;


void callback1_Func(int event, int x, int y, int flags, void* userdate);
void callback2_Func(int event, int x, int y, int flags, void* userdate);

class Part_3
{

    public:
    // members
    vector<Point2f> pos1,pos2;
    vector<Point2f> n_pos1,n_pos2;

    Mat img1, img2, undist_img1, undist_img2;
    double fx, fy, cx, cy;
    Mat dc, cam_mat, E, F, mask;

    // constructor
    Part_3()
    {
        dc = Mat::eye(5, 1, CV_64F);
        cam_mat = Mat::eye(3, 3, CV_64F);
    }

    // member functions
    void init_image(string a, string b)
    {
        img1 = imread(a);
        img2 = imread(b);
    }

    void display_image(Mat i1, Mat i2)
    {
        namedWindow("image1", WINDOW_NORMAL);
        namedWindow("image2", WINDOW_NORMAL);

        imshow("image1", i1);
        imshow("image2", i2);

        // set the callback function for any mouse event
        setMouseCallback("image1", callback1_Func, NULL);
        setMouseCallback("image2", callback2_Func, NULL);

        // Wait until user press some key
         waitKey(0);
    }

    //Mat kp1, kp2;
    //void detect_key_points(Mat i1, Mat i2)
    //{
    //    SiftFeatureDetector Detector;
    //    //SIFT Detector;
    //    Detector.detect(i1, kp1);
    //    Detector.detect(i2, kp2);
    //}

    void undistort_image()
    {
        string filename = "camera.yml";
        FileStorage fs(filename, FileStorage::READ);

        FileNode n = fs["camera_matrix"];
        FileNode ns = n["data"];
        fx = (double) ns[0];
        fy = (double) ns[4];
        cx = (double) ns[2];
        cy = (double) ns[5];

        FileNode d = fs["distortion_coefficients"];
        FileNode ds = d["data"];

        // distortion coefficient vector
        int k = 0;

        for(int r=0;r<5;r++)
            for(int q=0;q<1;q++)
            {
                dc.at<double>(r, q) = (double) ds[k];
                k++;
            }


        Mat lr, dt1, dt2;

        // camera matrix
        k = 0;
        for(int r=0;r<3;r++)
        {
            for(int q=0;q<3;q++)
            {
                cam_mat.at<double>(r, q) = (double) ns[k];
                k++;
            }
        }

        dt1 = getOptimalNewCameraMatrix(cam_mat, dc, img1.size(), 1.0, img1.size());
        dt2 = getOptimalNewCameraMatrix(cam_mat, dc, img2.size(), 1.0, img2.size());

        undist_img1 = img1.clone();
        undist_img2 = img2.clone();
        
        undistort(img1, undist_img1, dt1, dc);
        undistort(img2, undist_img2, dt2, dc);

        display_image(undist_img1, undist_img2);
    }

    void epipolar_image()
    {
        Point2d pp(cx, cy);
        //E = findEssentialMat(pos1, pos2, fx, pp, RANSAC, 0.999, 1.0, mask);
        //F = findFundamentalMat(pos1, pos2, CV_FM_RANSAC, 0.999, 1.0, mask);
        //cout << "E "<< E <<endl;
        //cout << "F "<< F <<endl;
        //cout << "Mask "<< (int) mask.at<char>(0,1) <<endl;
        
        //Mat H = findHomography(pos1, pos2, RANSAC, 5.0, mask);
        //cout << "H "<< H << endl;
        Mat me1, st1, me2, st2;
        //convertPointsToHomogeneous(pos1, pos1);
        //convertPointsToHomogeneous(pos2, pos2);
        meanStdDev(pos1, me1, st1);
        meanStdDev(pos2, me2, st2);

        //cout << "Mask "<< mask << endl;
        cout << "mean " << me1.at<double>(0,0) << endl;
        //cout << "std " << st << endl;

        for(int r = 0; r < pos1.size(); r++)
        {
            n_pos1.push_back(Point(abs(pos1[r].x - me1.at<double>(0,0))/fx, abs(pos1[r].y - me1.at<double>(0,1))/fy));
            n_pos2.push_back(Point(abs(pos2[r].x - me2.at<double>(0,0))/fx, abs(pos2[r].y - me2.at<double>(0,1))/fy));

            //n_pos1[r].y = pos1[r].y - me.at<double>(0,1);
        }

        meanStdDev(n_pos1, me1, st1);
        meanStdDev(n_pos2, me1, st2);
        cout << "s " << 1/st1.at<double>(0,0) <<endl;

        for(int r = 0; r < n_pos1.size(); r++)
        {
            n_pos1[r].x = n_pos1[r].x*1/st1.at<double>(0,0)*fx;
            n_pos1[r].y = n_pos1[r].y*1/st2.at<double>(0,1)*fy;
            n_pos2[r].x = n_pos2[r].x*1/st1.at<double>(0,0)*fx;
            n_pos2[r].y = n_pos2[r].y*1/st2.at<double>(0,1)*fy;
        }

        //for(int r = 0; r < n_pos1.size(); r++)
        //{
        //    n_pos1[r].x = n_pos1[r].x*fx;
        //    n_pos1[r].y = n_pos1[r].y*fy;
        //    n_pos2[r].x = n_pos2[r].x*fx;
        //    n_pos2[r].y = n_pos2[r].y*fy;
        //}

        //convertPointsFromHomogeneous(n_pos1, n_pos1);       
        //convertPointsFromHomogeneous(n_pos2, n_pos2);
        Size mask_size = mask.size();

        Mat epilines1, epilines2;
        //for(int r=0;r< mask_size.height;r++)
        //{
        //    //cout << "Mask num "<< mask.at<int>(r,0) << endl;
        //    if((int) mask.at<char>(r,0) == 1)
        //    {
        //        //cout << "Point "<< pos1[r].x <<endl;
        //        n_pos1.push_back(Point(pos1[r].x, pos1[r].y));
        //        n_pos2.push_back(Point(pos2[r].x, pos2[r].y));
        //    }
        //}
        
        //F = findFundamentalMat(n_pos1, n_pos2, CV_FM_8POINT, 0.0, 0.0);
        F = findFundamentalMat(n_pos1, n_pos2, CV_FM_RANSAC, 0.999, 1.0, mask);


        //Moments m = moments(mask, true);
        //cout << F << endl;
        //Point center(m.m10/m.m00, m.m01/m.m00);
        //cout << "Center "<< center <<endl;
        //n_pos1 = pos1;
        //n_pos2 = pos2;
        computeCorrespondEpilines(n_pos1, 1, F, epilines1); //Index starts with 1
        computeCorrespondEpilines(n_pos2, 2, F, epilines2);
        //cout << epilines1 <<endl;
        cout << n_pos1.size() << endl;
        
        //Mat h = findHomography(pos1, pos2);
        //warpPerspective(undist_img1, undist_img2, h, undist_img1.size());
        namedWindow("image1", WINDOW_NORMAL);
        namedWindow("image2", WINDOW_NORMAL);
        for(int r=0; r<n_pos1.size(); r++)
        {
            line(undist_img2, Point(0,-epilines1.at<float>(r,2)/epilines1.at<float>(r,1)), Point(undist_img2.cols,-(epilines1.at<float>(r,2)+epilines1.at<float>(r,0)*undist_img2.cols)/epilines1.at<float>(r,1)),Scalar(0,255,0), 5, CV_AA);
            line(undist_img1, Point(0,-epilines2.at<float>(r,2)/epilines2.at<float>(r,1)), Point(undist_img1.cols,-(epilines2.at<float>(r,2)+epilines2.at<float>(r,0)*undist_img1.cols)/epilines2.at<float>(r,1)),Scalar(0,255,0), 5, CV_AA);

        }
        //line(undist_img2, Point(pos1[0].x,pos1[0].y), Point(pos1[0].x + 2000,pos1[0].y + 2000), Scalar(0, 0, 255), 5,CV_AA);
        imshow("image1", undist_img1);
        imshow("image2", undist_img2);
        waitKey(0);
    }

}p;

// mouse callback function
void callback1_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {
        static int i=0;
        if (i >= 20) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        p.pos1.push_back(Point(x, y));
        i++;
        namedWindow("image1", WINDOW_NORMAL);
        rectangle(p.undist_img1, Point(x-50, y-50), Point(x+50, y+50), Scalar(0,255,0), -1);
        imshow("image1", p.undist_img1);
        waitKey(0);
    }
}

void callback2_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {   static int j=0;
        if (j >= 20) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        p.pos2.push_back(Point(x, y));
        j++;
        rectangle(p.undist_img2, Point(x-50, y-50), Point(x+50, y+50), Scalar(0,255,0), -1);
        namedWindow("image2",WINDOW_NORMAL);
        imshow("image2", p.undist_img2);
        waitKey(0);
    }
}


int main(int argc, char** argv)
{
    string a, b;
    a = "IMG_L.JPG";
    b = "IMG_C.JPG";
    p.init_image(a, b);
    p.undistort_image();
    p.epipolar_image();
    
    return 0;    
}
