#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face.hpp"
#include"opencv2/face/facerec.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ',') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	string fn_haar = "C:\\Program Files\\opencv\\install\\etc\\haarcascades\\haarcascade_frontalface_default.xml";
	//string fn_haar = "C:\\Program Files\\opencv\\install\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";

	string fn_csv = "C:\\Users\\ahang\\Pictures\\opencv\\faces\\face.csv";
	int deviceId = 1;
	vector<Mat> images;
	vector<int> labels;	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	//Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	Ptr<cv::face::BasicFaceRecognizer> model = face::createEigenFaceRecognizer();
	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	//model->load("D:\\ahang\\Pictures\\opencv\\faces\\faceasd");
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	VideoCapture cap;
	cap.open(deviceId);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 480);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 270);
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	Mat frame;
	while (1) {
		cap >> frame;
		resize(frame,frame,Size(480, 270));
		Mat original = frame.clone();
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);
		// At this point you have the position of the faces in
		// faces. Now we'll get the faces, make a prediction and
		// annotate it in the video. Cool or what?
		//找出图中最大的人脸
		int max = 0;
		int index = -1;
		for (int j = 0; j < faces.size(); j++)
		{
			if (faces[j].area() > max)
			{
				max = faces[j].area();
				index = j;
			}
		}
		if (index != -1)
		{
			Rect face_i = faces[index];
			int length = min(face_i.height, face_i.width);
			Mat face = gray(Rect(face_i.x, face_i.y, length, length));
			Mat face_resized;
			cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			imshow("resize", face_resized);
			int label;
			double confidence;
			model->predict(face_resized, label, confidence);
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			string box_text;
			if (label >= 1 && label <= 15)
				box_text = format("Prediction = %d, con = %lf,  Who are you ? ? ? ", label, confidence);
			else
				box_text = format("Prediction = %d, con = %lf", label, confidence);
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
			imshow("face_recognizer", original);

		}
		else
		{
			putText(original, "No Face!", Point(original.cols / 2, original.rows / 2), FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0, 255, 0), 2.0);
		}
		imshow("face_recognizer", original);
		char key = (char)waitKey(30);
		if (key == 27)
			break;
	}
	return 0;
}