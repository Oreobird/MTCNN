#include "mtcnn.h"
#include <unistd.h>
#include <stdint.h>
#include "timing.h"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>


using namespace std;
using namespace cv;

static void draw_mark(Mat inputImage, std::vector<Bbox> finalBbox)
{
    int left_eye_x = 0;
    int left_eye_y = 0;
    int right_eye_x = 0;
    int right_eye_y = 0;
    int nose_x = 0;
    int nose_y = 0;
    int mouth_left_x = 0;
    int mouth_left_y = 0;
    int mouth_right_x = 0;
    int mouth_right_y = 0;

	int num_box = finalBbox.size();
	printf("face num: %u \n", num_box);

    for (int i = 0; i < num_box; i++) {
        rectangle(inputImage, 
                    cvPoint(finalBbox[i].x1, finalBbox[i].y1),
                    cvPoint(finalBbox[i].x2, finalBbox[i].y2),
                    Scalar(255, 0, 0));
        left_eye_x = (int) (finalBbox[i].ppoint[0] + 0.5f);
        left_eye_y = (int) (finalBbox[i].ppoint[5] + 0.5f);
        right_eye_x = (int) (finalBbox[i].ppoint[1] + 0.5f);
        right_eye_y = (int) (finalBbox[i].ppoint[6] + 0.5f);

        circle(inputImage,
                cvPoint(left_eye_x, left_eye_y),
                3,
                Scalar(255, 0, 0));
        circle(inputImage,
                cvPoint(right_eye_x, right_eye_y),
                3,
                Scalar(255, 0, 0));

        nose_x = (int) (finalBbox[i].ppoint[2] + 0.5f);
        nose_y = (int) (finalBbox[i].ppoint[7] + 0.5f);
        circle(inputImage,
                cvPoint(nose_x, nose_y),
                3,
                Scalar(255, 0, 0));
        mouth_left_x = (int) (finalBbox[i].ppoint[3] + 0.5f);
        mouth_left_y = (int) (finalBbox[i].ppoint[8] + 0.5f);
        mouth_right_x = (int) (finalBbox[i].ppoint[4] + 0.5f);
        mouth_right_y = (int) (finalBbox[i].ppoint[9] + 0.5f);
        circle(inputImage,
                cvPoint(mouth_left_x, mouth_left_y),
                3,
                Scalar(255, 0, 0));
        circle(inputImage,
                cvPoint(mouth_right_x, mouth_right_y),
                3,
                Scalar(255, 0, 0));
    }
    printf("draw_mark done\n");
    imshow("detect", inputImage);
    //waitKey(0);
}


void detect(MTCNN mtcnn, Mat inputImage)
{
    int width = inputImage.cols;
    int height = inputImage.rows;
	std::vector<Bbox> finalBbox;

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(inputImage.data, ncnn::Mat::PIXEL_BGR2RGB, width, height);

	double startTime = now();

	mtcnn.detect(ncnn_img, finalBbox);

	double nDetectTime = calcElapsed(startTime, now());	
    printf("time: %d ms.\n ", (int)(nDetectTime * 1000));

    draw_mark(inputImage, finalBbox);
}


int main(int argc, char **argv) 
{
	const char *model_path = argv[1];
	char *szfile = argv[2];
	int Width = 0;
	int Height = 0;
    bool from_video = true;	
    VideoCapture capture;
    Mat inputImage;

	printf("mtcnn face detection\n");

	if (argc < 1) {
		printf("usage: %s  model_path image_file \n ", argv[0]);
		printf("eg: %s  ../models ../sample.jpg \n ", argv[0]);
		printf("press any key to exit. \n");
		getchar();
		return 0;
	}

    if (from_video)
    {
        capture.open(0); 
        int count = 1;
        if (capture.isOpened())
        {
            namedWindow("detect", WINDOW_KEEPRATIO);
            cout << "Capture is opened" << endl;
            int n = 0;
            char filename[200];

            while (true)
            {
                capture >> inputImage;
                if (inputImage.empty())
                    break;

	            MTCNN mtcnn(model_path);

                int width = inputImage.cols;
                int height = inputImage.rows;
	            std::vector<Bbox> finalBbox;

	            ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(inputImage.data, ncnn::Mat::PIXEL_BGR2RGB, width, height);

	            double startTime = now();

	            mtcnn.detect(ncnn_img, finalBbox);

	            double nDetectTime = calcElapsed(startTime, now());	
                printf("time: %d ms.\n ", (int)(nDetectTime * 1000));

                draw_mark(inputImage, finalBbox);

                char key = (char)waitKey(30);
                switch (key)
                {
                    case 'q':
                    case 'Q':
                    case 27:
                        return 0;
                    case ' ':
                        sprintf(filename, "face_%d.jpg", n++);
                        imwrite(filename, inputImage);
                        printf("Save %s\n", filename);
                        break;
                    default:
                        break;
                }
            }
        }
    }
    else
    {
	    MTCNN mtcnn(model_path);
        Mat iImage = imread(szfile);
        printf("inputImage width:%d, height:%d\n", iImage.cols, iImage.rows);
        int width = iImage.cols;
        int height = iImage.rows;
	    std::vector<Bbox> finalBbox;

	    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(iImage.data, ncnn::Mat::PIXEL_BGR2RGB, width, height);

	    double startTime = now();

	    mtcnn.detect(ncnn_img, finalBbox);

	    double nDetectTime = calcElapsed(startTime, now());	
        printf("time: %d ms.\n ", (int)(nDetectTime * 1000));

        draw_mark(iImage, finalBbox);
    }

    return 0;
}
