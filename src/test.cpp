#include <iostream>
#include "MNN_LFFD.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <chrono>
using namespace cv;

#define IMAGE_TEST
//#define VIDEO

int main(int argc, char** argv) {
	
    if (argc !=3)
	{
		std::cout << " name.exe mode_path image_file/video_file" << std::endl;
		return -1;
	}
	
	//eg :  std::string model_path="your_path/models"
	
	std::string model_path = argv[1];
	std::string image_or_video_file = argv[2];
	
	
	LFFD* face_detector = new LFFD(model_path,5,2);
	
#ifdef IMAGE_TEST
	cv::Mat image = cv::imread(image_or_video_file);
	std::vector<FaceInfo > finalBox;

	std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
	face_detector->detect(image, finalBox,image.rows,image.cols);
	std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
	std::cout << "mtcnn time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000 << "ms" << std::endl;

	for (int i = 0; i < finalBox.size(); i++) {
		FaceInfo facebox = finalBox[i];
		cv::Rect box=cv::Rect(facebox.x1,facebox.y1,facebox.x2-facebox.x1,facebox.y2-facebox.y1);
		cv::rectangle(image, box, cv::Scalar(255, 0, 21), 2);
	}
	std::cout << finalBox.size() << std::endl;
	cv::imwrite("res.jpg", image);
	cv::namedWindow("MNN", CV_WINDOW_NORMAL);
	cv::imshow("MNN", image);
	cv::waitKey();
#endif // IMAGE_TEST

#ifdef VIDEO
    cv::VideoCapture cap(image_or_video_file);
	cv::Mat image;

	int MAXNUM = 100;
	float AVE_TIME = 0;
	float MAX_TIME = -1;
	float MIN_TIME = 99999999999999;
	int count = 0;
	while (count<10000) {
		cap >> image;
		std::vector<FaceInfo > finalBox;
		std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
		face_detector->detect(image, finalBox);
		std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
		float dur = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000;
		std::cout << "lffd time:" << dur << "ms" << std::endl;

		AVE_TIME += dur;

		if (MAX_TIME < dur) {
			MAX_TIME = dur;
		}
		if (MIN_TIME > dur) {
			MIN_TIME = dur;
		}


		for (int i = 0; i < finalBox.size(); i++) {
			FaceInfo facebox = finalBox[i];
			cv::Rect box = cv::Rect(facebox.x1, facebox.y1, facebox.x2 - facebox.x1, facebox.y2 - facebox.y1);
			cv::rectangle(image, box, cv::Scalar(255, 0, 21), 2);
		}

	
		cv::namedWindow("MNN", CV_WINDOW_NORMAL);
		cv::imshow("MNN", image);
		cv::waitKey(1);
		count++;
	}

	std::cout << "IMAGE SHAPE: " << "640x480" << std::endl;
	std::cout << "MAX LOOP TIMES: " << MAXNUM << std::endl;
	std::cout << "AVE TIME: " << AVE_TIME / count << " ms" << std::endl;
	std::cout << "MAX TIME: " << MAX_TIME << " ms" << std::endl;
	std::cout << "MIN TIME: " << MIN_TIME << " ms" << std::endl;

#endif
	return 0;
}

