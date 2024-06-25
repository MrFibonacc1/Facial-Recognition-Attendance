#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <vector>
#include <string>
#include <map>

using namespace std;
using namespace cv;
using namespace cv::face;

void captureUserImages(const string& userName, vector<Mat>& images, vector<int>& labels, int label, CascadeClassifier& faceDetector) {
    cout << "Capturing 20 images for " << userName << ". Please look at the camera." << endl;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Camera not accessible" << endl;
        return;
    }

    Mat frame, grayFrame;
    vector<Rect> faces;
    int imagesCaptured = 0;

    while (imagesCaptured < 20) {
        cap >> frame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);
        faceDetector.detectMultiScale(grayFrame, faces, 1.3, 4, 0 | CASCADE_SCALE_IMAGE, Size(60, 60));

        for (const auto& face : faces) {
            Mat faceROI = grayFrame(face).clone();
            resize(faceROI, faceROI, Size(128, 128));

            images.push_back(faceROI);
            labels.push_back(label);
            imagesCaptured++;
            imshow("Capturing User Images", frame);
            waitKey(500); // 0.5 second delay
        }
    }
    cap.release();
    destroyWindow("Capturing User Images");
}

int main() {
    vector<Mat> images;
    vector<int> labels;
    map<string, int> students;
    map<int, string> labelToNameMap;
    string foundName = "";
    int labelCount = 0;

    Ptr<EigenFaceRecognizer> recognizer = EigenFaceRecognizer::create();
    string modelPath = "/Users/huzaifarehan/Desktop/Projects/C++-Workplace/face-detection Project/model.yml";

    CascadeClassifier faceDetector;
    if (!faceDetector.load("/Users/huzaifarehan/Desktop/Projects/C++-Workplace/face-detection Project/haarcascade_frontalface_default.xml")) {
        cerr << "Error loading Haar Cascade file." << endl;
        return -1;
    }

    int numUsers;
    cout << "Enter the number of users: ";
    cin >> numUsers;
    cin.ignore();

    for (int i = 0; i < numUsers; ++i) {
        string userName;
        students[userName] = 0;
        cout << "Enter name for user " << i + 1 << ": ";
        getline(cin, userName);
        labelToNameMap[labelCount] = userName;
        captureUserImages(userName, images, labels, labelCount++, faceDetector);
    }

    if (!images.empty()) {
        recognizer->train(images, labels);
        recognizer->save(modelPath);
        cout << "Model trained and saved as " << modelPath << endl;
    } else {
        cerr << "Error: No images captured for training." << endl;
        return 1;
    }

    VideoCapture video(0);

    while (true) {
        Mat frame, grayFrame;
        video.read(frame);
        rectangle(frame, Point(0,0), Point(250,70), Scalar(50, 50, 255), FILLED);
        putText(frame, foundName+" was found", Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 1);
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);

        vector<Rect> faces;
        faceDetector.detectMultiScale(grayFrame, faces, 1.2, 4, 0 | CASCADE_SCALE_IMAGE, Size(60, 60));

        for (size_t i = 0; i < faces.size(); i++) {
            Mat faceROI = grayFrame(faces[i]);
            if (!faceROI.isContinuous()) {
                faceROI = faceROI.clone();
            }
            resize(faceROI, faceROI, Size(128, 128));

            int label;
            double confidence;
            recognizer->predict(faceROI, label, confidence);

            string name = "Unknown";
            if (confidence < 5000 && labelToNameMap.find(label) != labelToNameMap.end()) {
                name = labelToNameMap[label];
                if(students[name]==0){
                    cout<<name<<" was found"<<endl;
                    students[name]=1;
                    foundName = name;
                }
            }

            putText(frame, name, faces[i].tl(), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
            rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(0, 255, 0), 2);
        }

        imshow("Frame", frame);
        if (waitKey(1) == 27) break; // Exit on ESC
    }

    video.release();
    destroyAllWindows();
    for (const auto& pair : students) {
        if(pair.second==0){
            cout<<pair.first<<" was not found" << endl;
        }
    }
    return 0;
}
