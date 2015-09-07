//http://opencv.jp/opencv-2.2/cpp/ml_support_vector_machines.html
//上記を参考に読んで理解する
//()のところで何を行っているか読む


//自分の環境に合わせて書き換える．
#include <opencv2\opencv.hpp>
#include <ml.h>
#include <time.h>


using namespace cv;

int
	main (int argc, char **argv)
{
	//出来たら数値を変えて実行してみる
	const int s = 1000;
	int size = 400;
	int i, j, sv_num;
	IplImage *img;
	CvSVM svm = CvSVM ();
	CvSVMParams param;
	CvTermCriteria criteria;
	CvRNG rng = cvRNG (time (NULL));
	CvPoint pts[s];
	float data[s * 2];
	int res[s];
	CvMat data_mat, res_mat;
	CvScalar rcolor;
	const float *support;

	// (1)画像領域の確保と初期化
	img = cvCreateImage (cvSize (size, size), IPL_DEPTH_8U, 3);
	cvZero (img);

	// (2)
	for (i = 0; i < s; i++) {
		pts[i].x = cvRandInt (&rng) % size;
		pts[i].y = cvRandInt (&rng) % size;
		
		//出来たら数値を変えて実行してみる
		if (pts[i].y > 200) {
			cvLine (img, cvPoint (pts[i].x - 2, pts[i].y - 2), cvPoint (pts[i].x + 2, pts[i].y + 2), CV_RGB (255, 0, 0));
			cvLine (img, cvPoint (pts[i].x + 2, pts[i].y - 2), cvPoint (pts[i].x - 2, pts[i].y + 2), CV_RGB (255, 0, 0));
			res[i] = 1;
		}
		else {
			cvLine (img, cvPoint (pts[i].x - 2, pts[i].y - 2), cvPoint (pts[i].x + 2, pts[i].y + 2), CV_RGB (0, 255, 0));
			cvLine (img, cvPoint (pts[i].x + 2, pts[i].y - 2), cvPoint (pts[i].x - 2, pts[i].y + 2), CV_RGB (0, 255, 0));
			res[i] = 2;
		}
	}

	// (3)学習データの表示
	cvNamedWindow ("SVM", CV_WINDOW_AUTOSIZE);
	cvShowImage ("SVM", img);
	cvWaitKey (0);

	// (4)
	for (i = 0; i < s; i++) {
		data[i * 2] = float (pts[i].x) / size;
		data[i * 2 + 1] = float (pts[i].y) / size;
	}
	cvInitMatHeader (&data_mat, s, 2, CV_32FC1, data);
	cvInitMatHeader (&res_mat, s, 1, CV_32SC1, res);
	criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams (CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);

	// (5)
	svm.train (&data_mat, &res_mat, NULL, NULL, param);

	// (6)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			CvMat m;
			float ret = 0.0;
			float a[] = { float (j) / size, float (i) / size };
			cvInitMatHeader (&m, 1, 2, CV_32FC1, a);
			ret = svm.predict (&m);
			switch ((int) ret) {
			case 1:
				rcolor = CV_RGB (100, 0, 0);
				break;
			case 2:
				rcolor = CV_RGB (0, 100, 0);
				break;
			case 3:
				rcolor = CV_RGB (0, 0, 100);
				break;
			}
			cvSet2D (img, i, j, rcolor);
		}
	}

	// (7)トレーニングデータの再描画
	for (i = 0; i < s; i++) {
		CvScalar rcolor;
		switch (res[i]) {
		case 1:
			rcolor = CV_RGB (255, 0, 0);
			break;
		case 2:
			rcolor = CV_RGB (0, 255, 0);
			break;
		case 3:
			rcolor = CV_RGB (0, 0, 255);
			break;
		}
		cvLine (img, cvPoint (pts[i].x - 2, pts[i].y - 2), cvPoint (pts[i].x + 2, pts[i].y + 2), rcolor);
		cvLine (img, cvPoint (pts[i].x + 2, pts[i].y - 2), cvPoint (pts[i].x - 2, pts[i].y + 2), rcolor);
	}

	// (8)
	sv_num = svm.get_support_vector_count ();
	for (i = 0; i < sv_num; i++) {
		support = svm.get_support_vector (i);
		cvCircle (img, cvPoint ((int) (support[0] * size), (int) (support[1] * size)), 5, CV_RGB (200, 200, 200));
	}

	// (9)画像の表示
	cvNamedWindow ("SVM", CV_WINDOW_AUTOSIZE);
	cvShowImage ("SVM", img);
	cvWaitKey (0);

	cvDestroyWindow ("SVM");
	cvReleaseImage (&img);

	return 0;
}
