#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/calib3d/calib3d.hpp"
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
        Mat mat_points_1 = Mat::ones(3,pos1.size(), CV_64F);
        Mat mat_points_2 = Mat::ones(3,pos1.size(), CV_64F);
        Point2d pp(cx, cy);
        E = findEssentialMat(pos1, pos2, fx, pp, RANSAC, 0.999, 1.0, mask);
        F = findFundamentalMat(pos1, pos2, CV_FM_RANSAC, 0.999, 1.0, mask);
        //cout << "E "<< E <<endl;
        //cout << "F "<< F <<endl;
        //cout << "Mask "<< (int) mask.at<char>(0,1) <<endl;

        Mat epilines1, epilines2, norm_epilines1, norm_epilines2;
        norm_epilines1 = Mat::zeros(8, 3, CV_64F);
        norm_epilines2 = Mat::zeros(8, 3, CV_64F);
        for(int r=0;r< pos1.size();r++)
        {
            //cout << "Mask num "<< mask.at<int>(r,0) << endl;
            if((int) mask.at<char>(r,0) == 1)
            {
                //cout << "Point "<< pos1[r].x <<endl;
                n_pos1.push_back(Point(pos1[r].x, pos1[r].y));
                n_pos2.push_back(Point(pos2[r].x, pos2[r].y));
            }
        }

       for(int s1 =0; s1<pos1.size(); s1++)
       {
            mat_points_1.at<double>(0,s1) = (double)n_pos1[s1].x;
            mat_points_1.at<double>(1,s1) = (double)n_pos1[s1].y;
        }

        for(int s2 =0; s2<pos1.size(); s2++)
       {
            mat_points_2.at<double>(0,s2) = (double)n_pos2[s2].x;
            mat_points_2.at<double>(1,s2) = (double)n_pos2[s2].y;
        }

           //transpose(F,F); 
           epilines1 = F*mat_points_2;
           //transpose(F,F);
           epilines2 = F*mat_points_1;

           transpose(epilines1, epilines1);
           transpose(epilines2, epilines2);
        computeCorrespondEpilines(n_pos1, 1, F, epilines1); //Index starts with 1
        computeCorrespondEpilines(n_pos2, 2, F, epilines2); 
    
        namedWindow("image1", WINDOW_NORMAL);
        namedWindow("image2", WINDOW_NORMAL);
        for(int r=0; r<n_pos1.size(); r++)
        {
            line(undist_img2, Point(0,-epilines1.at<float>(r,2)/epilines1.at<float>(r,1)), Point(undist_img2.cols,-(epilines1.at<float>(r,2)+epilines1.at<float>(r,0)*undist_img2.cols)/epilines1.at<float>(r,1)),Scalar(0,255,0), 5, CV_AA);
            line(undist_img1, Point(0,-epilines2.at<float>(r,2)/epilines2.at<float>(r,1)), Point(undist_img1.cols,-(epilines2.at<float>(r,2)+epilines2.at<float>(r,0)*undist_img1.cols)/epilines2.at<float>(r,1)),Scalar(0,255,0), 5, CV_AA);

        }
        imshow("image1", undist_img1);
        imshow("image2", undist_img2);
        waitKey(0);
    }

//Members for part_4
Mat U, V_t, Sig, U_t, V;
Mat R1, R2, W_t;
Mat W = (Mat_<double>(3,3)<< 0, -1, 0, 1, 0, 0, 0, 0, 1);
Mat r; double p_z;
vector<Point3f>point_in_world;
vector<Point3f>point_in_other_cam;
float Error;

vector<Point3f>pos1_hg, pos2_hg, pos1_hg_t;
double calculate_depth(Mat R, Mat r)
{

    Mat x_l, x_r, x_l_temp, x_r_temp, x_l_temp_t;   
    Mat pts_1, pts_2, result;
    double depth[n_pos1.size()];
    pts_1 = Mat::eye(3,8, CV_64F);
    pts_2 = Mat::eye(3,8, CV_64F);
    x_l = Mat::eye(2,8, CV_64F);
    x_r = Mat::eye(2,8, CV_64F);
    result = Mat::eye(1,8, CV_64F);
    int k = 0;
   
    convertPointsToHomogeneous(n_pos1, pos1_hg);
    convertPointsToHomogeneous(n_pos2, pos2_hg);
   
    // convert vector<point2f> to Mat
    for(int j =0;j < n_pos1.size(); j++)
    {
        pts_1.at<double>(0,j) = (double)pos1_hg[j].x;
        pts_1.at<double>(1,j) = (double)pos1_hg[j].y;
        pts_1.at<double>(2,j) = (double)pos1_hg[j].z;
    }
    for(int j =0;j < n_pos1.size(); j++)
    {
        pts_2.at<double>(0,j) = (double)pos2_hg[j].x;
        pts_2.at<double>(1,j) = (double)pos2_hg[j].y;
        pts_2.at<double>(2,j) = (double)pos2_hg[j].z;
    }

    //points in camera frame
    x_l_temp = cam_mat.inv()*pts_1;
    x_r_temp = cam_mat.inv()*pts_2;
    
    // converting back to euclidean
    for(int c =0; c<2; c++)
    {
        x_l.row(c) = (x_l_temp.row(c)+0);
        x_r.row(c) = (x_r_temp.row(c)+0);
    }

    //depth calculation
    for(int b =0; b < n_pos1.size(); b++)
    {
    result.col(b) =( -fx*((fx*r.col(0) - x_r.at<double>(0,b)*r.col(2)))/((fx*R.row(0) - x_r.at<double>(0,b)*(R.row(2)))*x_l_temp.col(b))); 
    depth[b] = result.at<double>(0);
    }
    return result.at<double>(0);

}



void compute_pairs(Mat& Rot, Mat& trans)
{
    SVD::compute(E, Sig, U, V_t);
    transpose(V_t, V);
    //V = V_t;
    if(determinant(U)==-1 &&determinant(V) ==-1)
    {
        cout<<"incorrect essential matrix"<<endl;
        return;
    }

    else if(determinant(U)==-1 ||determinant(V)==-1)
    {
        cout<<"inverted essential"<<endl;
        E = -E;
    }
    
    else
    {
        cout<<"Correct essential"<<endl;
    }

    SVD::compute(E, Sig, U, V_t);
    transpose(U,U_t);
    transpose(W,W_t);
   
    //cout<<"Sig " <<Sig<<endl;
    R1= U*W*V_t;
    R2=U*W_t*V_t;
    //cout << "U " << U << endl;
    //cout << "U row/col" << U.col(2) << endl;
    U.col(2).copyTo(r);
    transpose(r, r);
    int d;
    p_z = calculate_depth(R1,r);
    if(p_z<0)
    {   
        p_z = calculate_depth(R1,-r);
        if(p_z<0)
        {
            
            p_z = calculate_depth(R2,r);
            if(p_z<0)
            {
                
                p_z=calculate_depth(R2,-r);
                if(p_z<0)
                {
                    cout<<"incorrect essential matrix";
                    return;
                }
                else
                {
                    R2.copyTo(Rot);
                    r=-r;
                    r.copyTo(trans);
                }
            } 
            else
             {
                 R2.copyTo(Rot);
                 r.copyTo(trans);
             }
         }
        else
        {
            R1.copyTo(Rot);
            r=-r;
            r.copyTo(trans);
        }
    }
    else
    {
        R1.copyTo(Rot);
        r.copyTo(trans);
    }
   // cout<<"actual depth is "<<calculate_depth(R2,-r)<<" "<< calculate_depth(R2,r)<<" "<<calculate_depth(R1,r)<<" "<< calculate_depth(R1,-r) << endl;
  
    return;
}

void reprojection_errors(std::vector<Vec2f> imagepoints)
{
Mat rot, trans;
Mat cam_mat, dc;
dc = Mat::eye(5, 1, CV_64F);
cam_mat = Mat::eye(3, 3, CV_64F);
//imagepoints = Mat::eye(8,2, CV_64F);
FileStorage fs("camera.yml", FileStorage::READ);

        FileNode n = fs["camera_matrix"];
        FileNode ns = n["data"];
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
compute_pairs(rot, trans);
cout<<Mat(pos1_hg);
cout<<rot<<endl<<trans<<endl;
projectPoints(Mat(pos1_hg), rot, trans, cam_mat, dc, imagepoints); 
cout<<"reprojected points"<<Mat(imagepoints);

}



void first_image_pair(Mat &rotation, Mat&translation)
{
    string a, b;
    a = "IMG_L.JPG";
    b = "IMG_C.JPG";
    init_image(a, b);
    undistort_image();
    epipolar_image();
    compute_pairs(rotation, translation);
    pos1.clear();
    pos2.clear();
    n_pos1.clear();
    n_pos2.clear();
}

void second_image_pair(Mat &rotation, Mat&translation)
{
    string a, b;
    a = "IMG_C.JPG";
    b = "IMG_R.JPG";
    init_image(a, b);
    undistort_image();
    epipolar_image();
    compute_pairs(rotation, translation);
    pos1.clear();
    pos2.clear();
    n_pos1.clear();
    n_pos2.clear();
}

void third_image_pair(Mat &rotation, Mat &translation)
{
    string a, b;
    a = "IMG_L.JPG";
    b = "IMG_R.JPG";
    init_image(a, b);
    undistort_image();
    epipolar_image();
    compute_pairs(rotation, translation);
    pos1.clear();
    pos2.clear();
    n_pos1.clear();
    n_pos2.clear();
}


///Part 5
void rescale_tran_vec(Mat &R_12, Mat &R_23, Mat &R_13, Mat &r_12, Mat &r_23, Mat &r_13)
{
    double beta, gamma;
    first_image_pair(R_12, r_12);
    second_image_pair(R_23, r_23);
    third_image_pair(R_13, r_13);
    cout<<"first trans vector"<<"\t"<<r_12<<endl;
    cout<<"second trans vector"<<"\t"<<r_23<<endl;
    cout<<"third trans vector"<<"\t"<<r_13<<endl;
    normalize(r_12, r_12, 1, 0, NORM_L2, -1);
    normalize(r_23, r_23, 1, 0, NORM_L2, -1);
    normalize(r_13, r_13, 1, 0, NORM_L2, -1);
    r_12 = 
}

}p;

// mouse callback function
void callback1_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {
        static int i=0;
        if (i >= 30) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
       // cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
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
        if (j >= 30) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
       // cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
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
     Mat R_12, R_13, R_23, r_12, r_23, r_13;
     p.rescale_tran_vec(R_12, R_23, R_23, r_12, r_23, r_13);
    //std::vector<Vec2f> points;
    //p.reprojection_errors(points);
    return 0;    
}
