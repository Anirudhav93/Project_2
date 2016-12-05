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

    Mat imgLC_pos1, imgLC_pos2;

    // constructor
    Part_3()
    {
        dc = Mat::eye(5, 1, CV_64F);
        cam_mat = Mat::eye(3, 3, CV_64F);
        imgLC_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
        imgLC_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);
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

    void display_rectangle(Mat i1, Mat i2)
    {
        namedWindow("image1", WINDOW_NORMAL);
        namedWindow("image2", WINDOW_NORMAL);

        for(int it = 0; it < n_pos1.size(); it++)
        {
            rectangle(i1, Point(n_pos1[it].x-50, n_pos1[it].y-50), Point(n_pos1[it].x+50, n_pos1[it].y+50), Scalar(0,255,0), -1);
        }
        //imshow("image1", i1);
        //waitKey(0);

        //namedWindow("image2", WINDOW_NORMAL);
        for(int it = 0; it < n_pos2.size(); it++)
        {
            rectangle(i2, Point(n_pos2[it].x-50, n_pos2[it].y-50), Point(n_pos2[it].x+50, n_pos2[it].y+50), Scalar(0,255,0), -1);
        }
        imshow("image1", i1);
        imshow("image2", i2);
        waitKey(0);
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
        for(int it=0;it< pos1.size();it++)
        {
            if((int) mask.at<char>(it,0) == 1)
            {
                n_pos1.push_back(Point(pos1[it].x, pos1[it].y));
                n_pos2.push_back(Point(pos2[it].x, pos2[it].y));
            }
        }

        for(int s1 =0; s1<n_pos1.size(); s1++)
        {
            mat_points_1.at<double>(0,s1) = (double)n_pos1[s1].x;
            mat_points_1.at<double>(1,s1) = (double)n_pos1[s1].y;
        }

        for(int s2 =0; s2<n_pos1.size(); s2++)
        {
            mat_points_2.at<double>(0,s2) = (double)n_pos2[s2].x;
            mat_points_2.at<double>(1,s2) = (double)n_pos2[s2].y;
        }

        //transpose(F,F); 
        //epilines1 = F*mat_points_2;
        //transpose(F,F);
        //epilines2 = F*mat_points_1;

        //transpose(epilines1, epilines1);
        //transpose(epilines2, epilines2);
        computeCorrespondEpilines(n_pos1, 1, F, epilines1); //Index starts with 1
        computeCorrespondEpilines(n_pos2, 2, F, epilines2); 
    
        cout << "Num of inliers " << n_pos1.size() << endl;
        
        display_rectangle(undist_img1, undist_img2);
        namedWindow("image1", WINDOW_NORMAL);
        namedWindow("image2", WINDOW_NORMAL);
        for(int it=0; it<n_pos1.size(); it++)
        {
            line(undist_img2, Point(0,-epilines1.at<float>(it,2)/epilines1.at<float>(it,1)), Point(undist_img2.cols,-(epilines1.at<float>(it,2)+epilines1.at<float>(it,0)*undist_img2.cols)/epilines1.at<float>(it,1)),Scalar(0,255,0), 5, CV_AA);
            line(undist_img1, Point(0,-epilines2.at<float>(it,2)/epilines2.at<float>(it,1)), Point(undist_img1.cols,-(epilines2.at<float>(it,2)+epilines2.at<float>(it,0)*undist_img1.cols)/epilines2.at<float>(it,1)),Scalar(0,255,0), 5, CV_AA);

        }
        imshow("image1", undist_img1);
        imshow("image2", undist_img2);
        waitKey(0);
    }


    //Members for part_4
    Mat U, V_t, Sig, U_t, V;
    Mat R1, R2, W_t;
    Mat W = (Mat_<double>(3,3)<< 0, -1, 0, 1, 0, 0, 0, 0, 1);
    Mat t; double p_z;
    //Mat Rot, trans;
    vector<Point3f>point_in_world;
    vector<Point3f>point_in_other_cam;
    float Error;

    vector<Point3f>pos1_hg, pos2_hg, pos1_hg_t;
    double calculate_depth(Mat R, Mat r)
    {

        Mat x_l, x_r, x_l_temp, x_r_temp, x_l_temp_t;   
        Mat pts_1, pts_2, result;
        double depth[n_pos1.size()];
        pts_1 = Mat::eye(3,n_pos1.size(), CV_64F);
        pts_2 = Mat::eye(3,n_pos1.size(), CV_64F);
        x_l = Mat::eye(2,n_pos1.size(), CV_64F);
        x_r = Mat::eye(2,n_pos1.size(), CV_64F);
        result = Mat::eye(1, n_pos1.size(),CV_64F);
        int k = 0;
   
        convertPointsToHomogeneous(n_pos1, pos1_hg);
        convertPointsToHomogeneous(n_pos2, pos2_hg);
       
        // convert vector<point2f> to Mat
        for(int it =0;it < n_pos1.size(); it++)
        {
            pts_1.at<double>(0,it) = (double)pos1_hg[it].x;
            pts_1.at<double>(1,it) = (double)pos1_hg[it].y;
            pts_1.at<double>(2,it) = (double)pos1_hg[it].z;
        }
        for(int it =0;it < n_pos1.size(); it++)
        {
            pts_2.at<double>(0,it) = (double)pos2_hg[it].x;
            pts_2.at<double>(1,it) = (double)pos2_hg[it].y;
            pts_2.at<double>(2,it) = (double)pos2_hg[it].z;
        }

        //points in camera frame
        x_l_temp = cam_mat.inv()*pts_1;
        x_r_temp = cam_mat.inv()*pts_2;
    
        // converting back to euclidean
        for(int it =0; it<2; it++)
        {
            x_l.row(it) = (x_l_temp.row(it)+0);
            x_r.row(it) = (x_r_temp.row(it)+0);
        }

        //depth calculation
        //transpose(x_l_temp, x_l_temp_t);
        for(int it =0; it < n_pos1.size(); it++)
        {
        result.col(it) =( -fx*((fx*r.row(0) - x_r.at<double>(0,it)*r.row(2)))/((fx*R.row(0) - x_r.at<double>(0,it)*(R.row(2)))*x_l_temp.col(it))); 
        }
        return result.at<double>(0);
    }



    void compute_pairs(Mat& Rot, Mat& trans)
    {
        SVD::compute(E, Sig, U, V_t);
        if(determinant(U)==-1 &&determinant(V_t) ==-1)
        {
            cout<<"incorrect essential matrix"<<endl;
            return;
        }

        else if(determinant(U)==-1 ||determinant(V_t)==-1)
        {
            cout<<"inverted essential"<<endl;
            E = -E;
        }
    
        else
        {
            cout<<"Correct essential"<<endl;
        }

        SVD::compute(E, Sig, U, V);
        transpose(V,V_t);
        transpose(W,W_t);
   
        R1= U*W*V;
        R2= U*W_t*V_t;
        U.col(2).copyTo(t);
        cout << "t " << t <<endl;
        int d;
        p_z = calculate_depth(R1,t);
        if(p_z<0)
        {   
            p_z = calculate_depth(R1,-t);
            if(p_z<0)
            {
            
                p_z = calculate_depth(R2,t);
                if(p_z<0)
                {
                
                    p_z=calculate_depth(R2,-t);
                    if(p_z<0)
                    {
                        cout<<"incorrect essential matrix";
                        return;
                    }
                    else
                    {
                        R2.copyTo(Rot);
                        t=-t;
                        t.copyTo(trans);
                    }
                } 
                else
                 {
                     R2.copyTo(Rot);
                     t.copyTo(trans);
                 }
             }
            else
            {
                R1.copyTo(Rot);
                t=-t;
                t.copyTo(trans);
            }
        }
        else
        {
            R1.copyTo(Rot);
            t.copyTo(trans);
        }
        //cout<<"actual depth is "<<calculate_depth(R2,-t)<<" "<< calculate_depth(R2,t)<<" "<<calculate_depth(R1,t)<<" "<< calculate_depth(R1,-t) << endl;
  
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

        for(int it=0;it<5;it++)
            for(int q=0;q<1;q++)
            {
                dc.at<double>(it, q) = (double) ds[k];
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

    
    //vector<Point2f> imgLC_pos1,imgLC_pos2;
    //Mat imgLC_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
    //Mat imgLC_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);

    void first_image_pair(Mat &rotation, Mat&translation)
    {
        string a, b;
        cout << "First Pair " << endl;
        a = "IMG_L.JPG";
        b = "IMG_C.JPG";
        init_image(a, b);
        undistort_image();
        epipolar_image();
        compute_pairs(rotation, translation);
        imgLC_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
        imgLC_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);

        //cout << "here " << imgLC_pos1 << endl;
        for(int it=0;it< n_pos1.size();it++)
        {
            //imgLC_pos1.push_back(Point(n_pos1[it].x, n_pos1[it].y));
            //imgLC_pos2.push_back(Point(n_pos2[it].x, n_pos2[it].y));
            //cout << "check 1" <<endl;
            imgLC_pos1.at<double>(0,it) = (double)n_pos1[it].x;
            imgLC_pos1.at<double>(1,it) = (double)n_pos1[it].y;
            //cout << "check 2" <<endl;
            //imgLC_pos2.at<double>(it,0) = (double)n_pos2[it].x;
            //imgLC_pos2.at<double>(it,1) = (double)n_pos2[it].y;
        }
        //cout << "imgLC " << imgLC_pos1 << endl;
        pos1.clear();
        pos2.clear();
        n_pos1.clear();
        n_pos2.clear();
    }

    //vector<Point2f> imgCR_pos1,imgCR_pos2;
    Mat imgCR_pos1, imgCR_pos2;

    void second_image_pair(Mat &rotation, Mat&translation)
    {
        string a, b;
        cout << "Second Pair " << endl;
        a = "IMG_C.JPG";
        b = "IMG_R.JPG";
        init_image(a, b);
        undistort_image();
        epipolar_image();
        compute_pairs(rotation, translation);
        imgCR_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
        imgCR_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);

        for(int it=0;it< n_pos1.size();it++)
        {
            //imgCR_pos1.push_back(Point(n_pos1[it].x, n_pos1[it].y));
            //imgCR_pos2.push_back(Point(n_pos2[it].x, n_pos2[it].y));
            imgCR_pos1.col(it) = (Mat_<double>(2,1)<< n_pos1[it].x, n_pos1[it].y);
            imgCR_pos2.col(it) = (Mat_<double>(2,1)<< n_pos2[it].x, n_pos2[it].y);
        }
        pos1.clear();
        pos2.clear();
        n_pos1.clear();
        n_pos2.clear();
    }

    //vector<Point2f> imgLR_pos1,imgLR_pos2;
    Mat imgLR_pos1, imgLR_pos2;

    void third_image_pair(Mat &rotation, Mat &translation)
    {
        string a, b;
        cout << "Third Pair " << endl;
        a = "IMG_L.JPG";
        b = "IMG_R.JPG";
        init_image(a, b);
        undistort_image();
        epipolar_image();
        compute_pairs(rotation, translation);
        imgLR_pos1 = Mat::eye(3, n_pos1.size(), CV_64F);
        imgLR_pos2 = Mat::eye(3, n_pos1.size(), CV_64F);

        for(int it=0;it< n_pos1.size();it++)
        {
            //imgLR_pos1.push_back(Point(n_pos1[it].x, n_pos1[it].y));
            //imgLR_pos2.push_back(Point(n_pos2[it].x, n_pos2[it].y));
            imgLR_pos1.col(it) = (Mat_<double>(3,1)<< n_pos1[it].x, n_pos1[it].y, 1);
            imgLR_pos2.col(it) = (Mat_<double>(3,1)<< n_pos2[it].x, n_pos2[it].y, 1);
        }
        pos1.clear();
        pos2.clear();
        n_pos1.clear();
        n_pos2.clear();
    }


    //Part 5
    //Members
    Mat R_12, R_13, R_23, r_12, r_23, r_13;
    Mat sum_r, nsum_r;
    Mat beta, gamma;
    double norm_sum_r, norm_nsum_r; 

    void rescale_tran_vec()
    {
        first_image_pair(R_12, r_12);
        second_image_pair(R_23, r_23);
        third_image_pair(R_13, r_13);
        cout<<"first rot matrix"<<"\n"<<R_12<<endl;
        cout<<"second rot matrix"<<"\n"<<R_23<<endl;
        cout<<"third rot matrix"<<"\n"<<R_13<<endl;
        //normalize(r_12, r_12, 1, 0, NORM_L2, -1);
        //normalize(r_23, r_23, 1, 0, NORM_L2, -1);
        //normalize(r_13, r_13, 1, 0, NORM_L2, -1);
        cout<<"first trans vector"<<"\n"<<r_12<<endl;
        cout<<"second trans vector"<<"\n"<<r_23<<endl;
        cout<<"third trans vector"<<"\n"<<r_13<<endl;

        Mat r_12_3 = R_23*r_12 + r_23;
        cout << "new R1 " <<"\n"<<r_12_3<<endl;
        sum_r = r_12_3 + r_23 + r_13;
        cout << "Sum of vectors " << "\n" << sum_r << endl;
        
        norm_sum_r = norm(sum_r, NORM_L2);
        cout << "Norm of above sum " << norm_sum_r << endl;

        Mat tr_12, tr_13, tr_23;
        transpose(r_12_3, tr_12);
        transpose(r_23, tr_23);
        beta = - (tr_12*r_13)/(tr_12*r_23);
        gamma = -(tr_23*r_13)/(tr_23*r_12_3);
        cout << "beta " << beta << endl;
        cout << "gamma " << gamma << endl;

        nsum_r = r_13 + r_23*beta + r_12_3*gamma;
        cout << "Sum of n vectors " << "\n" << nsum_r << endl;

        norm_nsum_r = norm(nsum_r, NORM_L2);
        cout << "Norm of above sum " << norm_nsum_r << endl;
    }

    // Part 6
    // Members
    Mat H, hom_imgLR, hom_imgCR, disp_LR;

    void plane_stereo()
    {
        double dist;
        Mat n_p = (Mat_<double>(1,3)<< 0, 0, -1);
        for (int it = 1; it <= 527; it++) //need to set as fx
        {
            dist = fx/it;
            H = R_13 - (r_13*n_p)/dist;
            //hom_imgLR = cam_mat*H*cam_mat.inv()*imgLR_pos2;
            //hom_imgLR = cam_mat*H.inv()*cam_mat.inv()*imgLR_pos2;
            //warpPerspective(img1, hom_imgLR, H, img2.size());
            //disp_LR = abs(hom_imgLR - imgLR_pos1);
            //disp_LR = abs(hom_imgLR - img1);
        }
        //cout << "disp " << disp_LR << endl;
        //cout << hom_imgLR << endl;
        
        Mat disp;
        //warpPerspective(img1, disp, H, img2.size());
        Ptr<StereoBM> sbm = StereoBM::create(16, 2);
        //cout << "here 1" <<endl;
        //sbm->SADWindowSize(9);
        sbm->setBlockSize(9);
        //cout << " here 2" <<endl;
        sbm->setNumDisparities(112);
        //cout << " here 3" <<endl;
        sbm->setPreFilterSize(5);
        sbm->setPreFilterCap(61);
        sbm->setMinDisparity(-39);
        sbm->setTextureThreshold(507);
        sbm->setUniquenessRatio(0);
        sbm->setSpeckleWindowSize(0);
        sbm->setSpeckleRange(8);
        sbm->setDisp12MaxDiff(1);
        //*sbm(img1, img2, disp);
        cout << "here end" << endl;
        Mat g_img1, g_img2;
        cvtColor(img1, g_img1, CV_BGR2GRAY);
        cvtColor(img2, g_img2, CV_BGR2GRAY);
        //img1.convertTo(g_img1, CV_8UC1);
        //img2.convertTo(g_img2, CV_8UC1);
        cout << "convert worked" << endl;
        sbm->compute(g_img1, g_img2, disp);
        cout << "here actual end" <<endl;

        namedWindow("disp_image", WINDOW_NORMAL);
        imshow("disp_image", disp);
        waitKey(0);

    }

}p;

// mouse callback function
void callback1_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {
        static int num_points1 =0;
        if (num_points1 >= 100) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
       // cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        p.pos1.push_back(Point(x, y));
        num_points1++;
        namedWindow("image1", WINDOW_NORMAL);
        rectangle(p.undist_img1, Point(x-50, y-50), Point(x+50, y+50), Scalar(0,0,255), -1);
        imshow("image1", p.undist_img1);
        waitKey(0);
    }
}

void callback2_Func(int event, int x, int y, int flags, void* userdate)
{
    if (event == EVENT_LBUTTONDOWN )
    {   static int num_points2=0;
        if (num_points2 >= 100) 
        {
            cout << "Too many points clicked." <<endl;
            return;
        }
       // cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
        p.pos2.push_back(Point(x, y));
        num_points2++;
        rectangle(p.undist_img2, Point(x-50, y-50), Point(x+50, y+50), Scalar(0,0,255), -1);
        namedWindow("image2",WINDOW_NORMAL);
        imshow("image2", p.undist_img2);
        waitKey(0);
    }
}


int main(int argc, char** argv)
{  
        //Ptr<StereoBM> sbm = StereoBM::create(16,2);
        //cout << "here 1" <<endl;
        //sbm->SADWindowSize = 9;
        //sbm->setBlockSize(9);
        //cout << " here 2" <<endl;
        //sbm->setNumDisparities(112);
        //cout << " here 3" <<endl;
        //sbm->setPreFilterSize(5);
        //sbm->setPreFilterCap(61);
        //sbm->setMinDisparity(-39);
        //sbm->setTextureThreshold(507);
        //sbm->setUniquenessRatio(0);
        //sbm->setSpeckleWindowSize(0);
        //sbm->setSpeckleRange(8);
        //sbm->setDisp12MaxDiff(1);
        //sbm->setDisp12MaxDiff(1);
        //cout << " here 4" <<endl;

        //sbm->setSpeckleRange(8);
        //sbm->setSpeckleWindowSize(0);
        //sbm->setUniquenessRatio(0);
        //sbm->setTextureThreshold(507);
        //sbm->setMinDisparity(-39);
        //sbm->setPreFilterCap(61);
        //sbm->setPreFilterSize(5);
    p.rescale_tran_vec();
    p.plane_stereo();
    //std::vector<Vec2f> points;
    //p.reprojection_errors(points);
    imwrite("epipolar_L.jpg", p.undist_img1);
    imwrite("epipolar_C.jpg", p.undist_img2);

    return 0;    
}
