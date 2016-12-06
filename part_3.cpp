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
    Point2d pp;
    vector<vector<Point3f> >objectPoints;
    vector<vector<Point2f> > imagePoints;
    Mat img1, img2, undist_img1, undist_img2;
    vector<Mat>rvecs;
    vector<Mat>tvecs;
    vector<float> per_view;
    double fx, fy, cx, cy;
    Mat dc, M,  cam_mat, E, F, mask;

    Mat imgLC_pos1, imgLC_pos2;

    int choose;
    char click; 

    // constructor
    Part_3()
    {
        dc = Mat::eye(5, 1, CV_64F);
        cam_mat = Mat::eye(3, 3, CV_64F);
        imgLC_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
        imgLC_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);
        choose = 1;
        click = 'y';
        pp.x =0.0;
        pp.y =0.0;
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


        if (click == 'y')
        {
            imshow("image1", i1);
            imshow("image2", i2);

        // set the callback function for any mouse event
            setMouseCallback("image1", callback1_Func, NULL);
            setMouseCallback("image2", callback2_Func, NULL);
        }
        else 
        {
        if (choose == 1)
        {
            pos1.push_back(Point(279, 537));
            pos1.push_back(Point(589, 418));
            pos1.push_back(Point(609, 1147));
            pos1.push_back(Point(785, 1002));
            pos1.push_back(Point(920, 997));
            pos1.push_back(Point(909, 1271));
            pos1.push_back(Point(821, 1333));
            pos1.push_back(Point(889, 1431));
            pos1.push_back(Point(1121, 1266));
            pos1.push_back(Point(1116, 1080));
            pos1.push_back(Point(1183, 976));
            pos1.push_back(Point(1338, 1070));
            pos1.push_back(Point(1390, 1318));
            pos1.push_back(Point(1493, 1178));
            pos1.push_back(Point(1597, 1178));
            pos1.push_back(Point(1225, 1674));
            pos1.push_back(Point(2264, 940));
            pos1.push_back(Point(1933, 594));

            pos2.push_back(Point(1423, 675));
            pos2.push_back(Point(1693, 555));
            pos2.push_back(Point(1704, 1210));
            pos2.push_back(Point(1834, 1080));
            pos2.push_back(Point(1943, 1075));
            pos2.push_back(Point(1906, 1324));
            pos2.push_back(Point(1849, 1392));
            pos2.push_back(Point(1891, 1480));
            pos2.push_back(Point(2093, 1324));
            pos2.push_back(Point(2099, 1148));
            pos2.push_back(Point(2286, 1137));
            pos2.push_back(Point(2171, 1049));
            pos2.push_back(Point(2338, 1371));
            pos2.push_back(Point(2421, 1241));
            pos2.push_back(Point(2530, 1236));
            pos2.push_back(Point(2151, 1709));
            pos2.push_back(Point(3283, 982));
            pos2.push_back(Point(2961, 628));
            //just for commit

            for(int it = 0; it < pos1.size(); it++)
            {
                rectangle(i1, Point(pos1[it].x-50, pos1[it].y-50), Point(pos1[it].x+50, pos1[it].y+50), Scalar(0,0,255), -1);
                rectangle(i2, Point(pos2[it].x-50, pos2[it].y-50), Point(pos2[it].x+50, pos2[it].y+50), Scalar(0,0,255), -1);

            }
            choose = 2;
        }

        else if (choose == 2)
        {
            pos1.push_back(Point(444, 847));
            pos1.push_back(Point(765, 672));
            pos1.push_back(Point(1426, 672));
            pos1.push_back(Point(1690, 553));
            pos1.push_back(Point(1700, 1204));
            pos1.push_back(Point(1199, 1338));
            pos1.push_back(Point(1829, 1085));
            pos1.push_back(Point(1943, 1080));
            pos1.push_back(Point(1912, 1323));
            pos1.push_back(Point(1850, 1385));
            pos1.push_back(Point(1897, 1478));
            pos1.push_back(Point(2093, 1333));
            pos1.push_back(Point(2093, 1142));
            pos1.push_back(Point(2176, 1049));
            pos1.push_back(Point(2295, 1142));
            pos1.push_back(Point(2352, 1369));
            pos1.push_back(Point(2434, 1250));
            pos1.push_back(Point(2150, 1721));
            pos1.push_back(Point(2961, 625));

            pos2.push_back(Point(1252, 883));
            pos2.push_back(Point(1527, 711));
            pos2.push_back(Point(2130, 670));
            pos2.push_back(Point(2400, 535));
            pos2.push_back(Point(2400, 1184));
            pos2.push_back(Point(1896, 1340));
            pos2.push_back(Point(2509, 1065));
            pos2.push_back(Point(2639, 1049));
            pos2.push_back(Point(2582, 1314));
            pos2.push_back(Point(2535, 1371));
            pos2.push_back(Point(2577, 1465));
            pos2.push_back(Point(2790, 1309));
            pos2.push_back(Point(2779, 1122));
            pos2.push_back(Point(2878, 1013));
            pos2.push_back(Point(2987, 1101));
            pos2.push_back(Point(3049, 1345));
            pos2.push_back(Point(3138, 1215));
            pos2.push_back(Point(2810, 1714));
            pos2.push_back(Point(3777, 535));
            //just commit

            for(int it = 0; it < pos1.size(); it++)
            {
                rectangle(i1, Point(pos1[it].x-50, pos1[it].y-50), Point(pos1[it].x+50, pos1[it].y+50), Scalar(0,0,255), -1);
                rectangle(i2, Point(pos2[it].x-50, pos2[it].y-50), Point(pos2[it].x+50, pos2[it].y+50), Scalar(0,0,255), -1);

            }
            choose = 3;
        }

        else if (choose == 3)
        {
            pos1.push_back(Point(284, 537));
            pos1.push_back(Point(594, 423));
            pos1.push_back(Point(609, 1142));
            pos1.push_back(Point(790, 1008));
            pos1.push_back(Point(920, 1008));
            pos1.push_back(Point(909, 1261));
            pos1.push_back(Point(821, 1333));
            pos1.push_back(Point(889, 1431));
            pos1.push_back(Point(1121, 1271));
            pos1.push_back(Point(1116, 1080));
            pos1.push_back(Point(1183, 971));
            pos1.push_back(Point(1333, 1070));
            pos1.push_back(Point(1390, 1318));
            pos1.push_back(Point(1499, 1188));
            pos1.push_back(Point(1592, 1188));
            pos1.push_back(Point(1235, 1685));
            pos1.push_back(Point(1809, 1281));
            pos1.push_back(Point(1933, 589));

            pos2.push_back(Point(2125, 670));
            pos2.push_back(Point(2400, 529));
            pos2.push_back(Point(2395, 1184));
            pos2.push_back(Point(2514, 1059));
            pos2.push_back(Point(2623, 1054));
            pos2.push_back(Point(2587, 1309));
            pos2.push_back(Point(2535, 1371));
            pos2.push_back(Point(2582, 1465));
            pos2.push_back(Point(2790, 1309));
            pos2.push_back(Point(2779, 1111));
            pos2.push_back(Point(2868, 1018));
            pos2.push_back(Point(2998, 1106));
            pos2.push_back(Point(3044, 1361));
            pos2.push_back(Point(3138, 1210));
            pos2.push_back(Point(3242, 1215));
            pos2.push_back(Point(2805, 1704));
            pos2.push_back(Point(3605, 1309));
            pos2.push_back(Point(3782, 535));
            //just commit

            for(int it = 0; it < pos1.size(); it++)
            {
                rectangle(i1, Point(pos1[it].x-50, pos1[it].y-50), Point(pos1[it].x+50, pos1[it].y+50), Scalar(0,0,255), -1);
                rectangle(i2, Point(pos2[it].x-50, pos2[it].y-50), Point(pos2[it].x+50, pos2[it].y+50), Scalar(0,0,255), -1);

            }
            choose = 1;
        }

        imshow("image1", i1);
        imshow("image2", i2);
        }

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
        
        //hacky correction to camera matrix
        invert(cam_mat, M);
        M = M*fx;


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
        cout << "E "<< E <<endl;
        cout << "F "<< F <<endl;
        //cout << "Mask "<< (int) mask.at<char>(0,1) <<endl;

     Mat epilines1, epilines2;
     for(int it=0;it< pos1.size();it++)
        {
            if((int) mask.at<char>(it,0) == 1)
            {
                n_pos1.push_back(Point(pos1[it].x, pos1[it].y));
                n_pos2.push_back(Point(pos2[it].x, pos2[it].y));
            }
        }

        

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
    Mat t;
    double p_z;
    Mat x_l, x_r, x_l_temp, x_r_temp, x_l_temp_t;

    float Error;
    vector<Point3f>x_l_gvec;
    vector<Point3f>pos1_hg, pos2_hg, pos1_hg_t;
    


    double calculate_depth(Mat R, Mat r)
    {
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
        recoverPose(E, n_pos1, n_pos2, Rot, trans);
        return;
    }


    void reprojection( Mat &rot, Mat &trans)
    {
    Mat imagepoints;
    Matx<double, 3, 1> trans_vec;
    Matx<double, 3, 1> rot_vec;
    vector<Point2f> d2_vector;

    Rodrigues(rot, rot_vec);
    trans_vec.operator()(0,0) = trans.at<double>(0,0);
    trans_vec.operator()(1,0) = trans.at<double>(1,0);
    trans_vec.operator()(2,0) = trans.at<double>(2,0);
    cout<<"camera coordinates"<<"\t"<<x_l_temp<<endl;
    cout<<"previously used:-"<<endl<<Mat(pos1_hg)<<endl;
    transpose(x_l_temp, x_l_temp_t);
    projectPoints(x_l_temp_t, rot, trans_vec, cam_mat, dc, imagepoints);
    cout<<"reprojected points"<<Mat(imagepoints);
    namedWindow("image1", WINDOW_NORMAL);
    namedWindow("image2", WINDOW_NORMAL);
    for(int it=0; it<n_pos1.size(); it++)
    {
        rectangle(undist_img2, Point(imagepoints.at<double>(it,0)-70, imagepoints.at<double>(it,1)-70), Point(imagepoints.at<double>(it,0)+70, imagepoints.at<double>(it,1)+70), Scalar(255, 0,0), -1);   
    }
    compute_pairs(rot, trans);
    cout<<Mat(pos1_hg);
    cout<<rot<<endl<<trans<<endl;
    projectPoints(Mat(pos1_hg), rot, trans, cam_mat, dc, imagepoints); 
    cout<<"reprojected points"<<Mat(imagepoints);

    imshow("image1", undist_img1);
    imshow("image2", undist_img2);
    waitKey(0);
    }

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
        reprojection(rotation, translation);
        imgLC_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
        imgLC_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);

        for(int it=0;it< n_pos1.size();it++)
        {
            imgLC_pos1.col(it) = (Mat_<double>(2,1)<< n_pos1[it].x, n_pos1[it].y);
            imgLC_pos2.col(it) = (Mat_<double>(2,1)<< n_pos2[it].x, n_pos2[it].y);
        }
        clear_vectors();
    }

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
        reprojection(rotation, translation);
        imgCR_pos1 = Mat::eye(2, n_pos1.size(), CV_64F);
        imgCR_pos2 = Mat::eye(2, n_pos1.size(), CV_64F);

        for(int it=0;it< n_pos1.size();it++)
        {
            imgCR_pos1.col(it) = (Mat_<double>(2,1)<< n_pos1[it].x, n_pos1[it].y);
            imgCR_pos2.col(it) = (Mat_<double>(2,1)<< n_pos2[it].x, n_pos2[it].y);
        }
        clear_vectors();
    }

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
        reprojection(rotation, translation);
        imgLR_pos1 = Mat::eye(3, n_pos1.size(), CV_64F);
        imgLR_pos2 = Mat::eye(3, n_pos1.size(), CV_64F);

        for(int it=0;it< n_pos1.size();it++)
        {
            imgLR_pos1.col(it) = (Mat_<double>(3,1)<< n_pos1[it].x, n_pos1[it].y, 1);
            imgLR_pos2.col(it) = (Mat_<double>(3,1)<< n_pos2[it].x, n_pos2[it].y, 1);
        }
        clear_vectors();
    }

void clear_vectors()
{
    pos1.clear();
    pos2.clear();
    x_l_gvec.clear();
    n_pos1.clear();
    n_pos2.clear();
}


    //Part 5
    //Members
    Mat R_12, R_13, R_23, r_12, r_23, r_13;
    Mat sum_r, nsum_r;
    Mat beta, gamma;
    double norm_sum_r, norm_nsum_r; 
    Mat r_12_1, r_23_1, r_13_1;
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

        r_12_1 = R_12*r_12;
        r_23_1 = R_13*r_23;
        r_13_1 = (-R_13)*r_13;
        sum_r = r_12_1 + r_23_1 + r_13_1;
        cout << "Sum of vectors " << "\n" << sum_r << endl;
        
        norm_sum_r = norm(sum_r, NORM_L2);
        cout << "Norm of above sum " << norm_sum_r << endl;

        Mat tr_12, tr_13, tr_23;
        transpose(r_12_1, tr_12);
        transpose(r_23_1, tr_23);
        //Mat beta1 = (tr_12*r_13);
        //Mat beta2 = (tr_13*r_23);
        //beta = -beta1/beta2;
        beta = - (tr_12*r_13_1)/(tr_23*r_13_1);
        //Mat gamma1 = (tr_12*r_23);
        //Mat gamma2 = (tr_13*r_23);
        //gamma = -gamma1/gamma2;
        gamma = -(tr_12*r_23_1)/(tr_23*r_13_1);
        cout << "beta " << beta << endl;
        cout << "gamma " << gamma << endl;

        nsum_r = r_12_1 + r_23_1*beta + r_13_1*gamma;
        cout << "Sum of n vectors " << "\n" << nsum_r << endl;

        norm_nsum_r = norm(nsum_r, NORM_L2);
        cout << "Norm of above sum " << norm_nsum_r << endl;
    }

    // Part 6
    // Members
    Mat H, Hm, hom_imgLC, hom_imgLR;
    Mat img_L, img_C, img_R;
    Mat disp_LC, disp_LR, filt_LC[530], filt_LR[530];
    Mat min_disp_LC, min_disp_LR;

    int max_ker_len, sad_size1, sad_size2;

    void plane_stereo()
    {
        double dist;
        Mat n_p = (Mat_<double>(1,3)<< 0, 0, -1);
        img_L = imread("IMG_L.JPG", 0);
        img_C = imread("IMG_C.JPG", 0);
        img_R = imread("IMG_R.JPG", 0);
        max_ker_len = 31;
        sad_size1 = 0;
        for (int it1 = 1; it1 <= (int) fx; it1+=50)
        {
            dist = fx/it1;
            H = R_12 - (r_12_1*n_p)/dist;
            Hm = cam_mat*H.inv()*cam_mat.inv();
            //hom_imgLC = Hm*img_C;
            warpPerspective(img_C, hom_imgLC, Hm, img_L.size());
            absdiff(img_L, hom_imgLC, disp_LC);
            for (int it2 = 1; it2 < max_ker_len; it2+=2) 
            {
                boxFilter(disp_LC, filt_LC[sad_size1], 10, Size(it2, it2), Point(-1,-1), false);
            }
            sad_size1++;
        }
        
        min_disp_LC = img_L.clone();
        for (int it1 = 0; it1 <= img_L.rows; it1++)
        {
            for (int it2 = 0; it2 <= img_L.cols; it2++)
            {
                min_disp_LC.at<double>(it1, it2) = filt_LC[0].at<double>(it1, it2);
                for (int it3 = 1; it3 < sad_size1; it3++)
                {
                    if (filt_LC[it3].at<double>(it1, it2) < min_disp_LC.at<double>(it1, it2))
                    {
                        min_disp_LC.at<double>(it1, it2) = filt_LC[it3].at<double>(it1, it2);
                    }
                }
            }
        }

        for (int it1 = 1; it1 <= (int) fx; it1+=50)
        {
            dist = fx/it1;
            H = R_13 - (r_13_1*n_p)/dist;
            Hm = cam_mat*H.inv()*cam_mat.inv();
            //hom_imgLR = Hm*img_R;
            warpPerspective(img_R, hom_imgLR, Hm, img_L.size());
            absdiff(img_L, hom_imgLR, disp_LR);
            for (int it2 = 1; it2 < max_ker_len; it2+=2) 
            {
                boxFilter(disp_LR, filt_LR[sad_size2], 10, Size(it2, it2), Point(-1,-1), false);
            }
            sad_size2++;
        }
        
        min_disp_LR = img_L.clone();
        for (int it1 = 0; it1 <= img_L.rows; it1++)
        {
            for (int it2 = 0; it2 <= img_L.cols; it2++)
            {
                min_disp_LR.at<double>(it1, it2) = filt_LR[0].at<double>(it1, it2);
                for (int it3 = 1; it3 < sad_size2; it3++)
                {
                    if (filt_LR[it3].at<double>(it1, it2) < min_disp_LR.at<double>(it1, it2))
                    {
                        min_disp_LR.at<double>(it1, it2) = filt_LR[it3].at<double>(it1, it2);
                    }
                }
            }
        }
        
        //Mat disp1, disp2;
        //Ptr<StereoBM> sbm = StereoBM::create(16, 2);
        //sbm->setBlockSize(9);
        //sbm->setNumDisparities(112);
        //sbm->setPreFilterSize(5);
        //sbm->setPreFilterCap(61);
        //sbm->setMinDisparity(-39);
        //sbm->setTextureThreshold(507);
        //sbm->setUniquenessRatio(0);
        //sbm->setSpeckleWindowSize(0);
        //sbm->setSpeckleRange(8);
        //sbm->setDisp12MaxDiff(1);
        //sbm->compute(img_L, img_C, disp1);
        //sbm->compute(img_L, img_R, disp2);

        destroyAllWindows();

        namedWindow("disp_image1", WINDOW_NORMAL);
        namedWindow("disp_image2", WINDOW_NORMAL);

        namedWindow("disp_image_f1", WINDOW_NORMAL);
        namedWindow("disp_image_f2", WINDOW_NORMAL);

        imshow("disp_image1", min_disp_LC);
        imshow("disp_image2", min_disp_LR);

        imshow("disp_image_f1", hom_imgLC);
        imshow("disp_image_f2", hom_imgLR);
        waitKey(0);

    }


///Members for part 7
Mat filt_LCR[530], min_disp_LCR;


void multi_plane_stereo()
{
    for (int it1 =0; it1<sad_size2; it1++)
    {
    add(filt_LC[it1], filt_LR[it1], filt_LCR[it1]);
    }
     min_disp_LCR = img_L.clone();
        for (int it1 = 0; it1 <= img_L.rows; it1++)
        {
            for (int it2 = 0; it2 <= img_L.cols; it2++)
            {
                min_disp_LCR.at<double>(it1, it2) = filt_LCR[0].at<double>(it1, it2);
                for (int it3 = 1; it3 < sad_size2; it3++)
                {
                    if (filt_LCR[it3].at<double>(it1, it2) < min_disp_LCR.at<double>(it1, it2))
                    {
                        min_disp_LCR.at<double>(it1, it2) = filt_LCR[it3].at<double>(it1, it2);


                    }
                }
            }
        }
 
        destroyAllWindows();

        namedWindow("disp_image_multi", WINDOW_NORMAL);
        //namedWindow("disp_image_f1", WINDOW_NORMAL);

        imshow("disp_image_multi", min_disp_LCR);

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
       cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;        
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
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
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
    cout << "Click on points?(y/n) ";
    cin >> p.click;
    cout << endl;
    p.rescale_tran_vec();
    p.plane_stereo();
    p.multi_plane_stereo();
    //std::vector<Vec2f> points;
    //p.reprojection_errors(points);
    //imwrite("epipolar_L.jpg", p.undist_img1);
    //imwrite("epipolar_C.jpg", p.undist_img2);

    return 0;    
}
