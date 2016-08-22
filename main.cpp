#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>
#include <array>
#include <math.h>
#include "nn.h"

using namespace std;
using namespace cv;

#define SHOW_EVERY_IMAGE 0
#define SHOW_ALL_IMAGES 0

int readImages(const char* fileName, vector<Mat>& images) {
    ifstream trainingData(fileName, ios::in|ios::binary);
    if (!trainingData.is_open()) {
        cout << "Failed to open file";
        return -1;
    }
    int32_t magic;
    trainingData.read((char*)&magic, sizeof(int32_t));
    magic = __builtin_bswap32(magic);
    if (magic != 2051) {
        return -1;
    }
    int32_t numberOfImages;
    trainingData.read((char*)&numberOfImages, sizeof(int32_t));
    numberOfImages = __builtin_bswap32(numberOfImages);
    if (numberOfImages <= 0) {
        return -1;
    }
    int32_t numberOfRow;
    int32_t numberOfColumns;
    trainingData.read((char*)&numberOfRow, sizeof(int32_t));
    trainingData.read((char*)&numberOfColumns, sizeof(int32_t));
    numberOfRow = __builtin_bswap32(numberOfRow);
    numberOfColumns = __builtin_bswap32(numberOfColumns);
    if (numberOfRow <= 0 || numberOfColumns <= 0) {
        return -1;
    }
    cout << "Number of images: " << numberOfImages << endl << "Size: " << numberOfRow << " X " << numberOfColumns << endl;
    images.reserve(numberOfImages);
    int imageSize = numberOfRow * numberOfColumns;
#if SHOW_ALL_IMAGES
    int lineSize = sqrt(numberOfImages);
    cout << "Line length: " << lineSize << endl;
    Mat allImages(lineSize * numberOfRow, lineSize * numberOfColumns, CV_8UC1);
    int imageColumnOffset = numberOfColumns * lineSize;
#endif
    for (int i = 0; i < numberOfImages; ++i) {
        Mat image(numberOfRow, numberOfColumns, CV_8UC1);
        trainingData.read((char*)image.data, imageSize);
        images.push_back(image);
#if SHOW_ALL_IMAGES
        int filledLines = i / lineSize;
        //calculate offset in the big matrix
        uchar* allImagesPtr = allImages.data +
                (filledLines * imageSize) * lineSize + (i - filledLines * lineSize) * numberOfColumns;
        for (int j = 0; j < numberOfRow; ++j) {
            memcpy(allImagesPtr, image.row(j).data, numberOfColumns);
            allImagesPtr += imageColumnOffset;
        }
#endif
#if SHOW_EVERY_IMAGE
        namedWindow("test image", WINDOW_AUTOSIZE);
        imshow("test image", image);
        waitKey(0);
#endif
    }
    trainingData.close();
#if SHOW_ALL_IMAGES
    namedWindow("test image", WINDOW_AUTOSIZE);
    imshow("test image", allImages);
    waitKey(0);
#endif
}

int readLabels(const char* fileName, vector<int8_t>& labels) {
    ifstream trainingLables(fileName, ios::in|ios::binary);
    if (!trainingLables.is_open()) {
        cout << "Failed to open file";
        return -1;
    }
    int32_t magic;
    trainingLables.read((char*)&magic, sizeof(int32_t));
    magic = __builtin_bswap32(magic);
    if (magic != 2049) {
        return -1;
    }
    int32_t numberOfLables;
    trainingLables.read((char*)&numberOfLables, sizeof(int32_t));
    numberOfLables = __builtin_bswap32(numberOfLables);
    if (numberOfLables <= 0) {
        return -1;
    }
    labels.reserve(numberOfLables);
    trainingLables.read((char*)labels.data(), numberOfLables);
    cout << "Lablels count: " << numberOfLables << endl;
}

int main(int argc, char *argv[]) {
    vector<int> config = {2, 3, 2, 1};
    NN net(config);
    net.traceConfig();
    Mat testInput = Mat::ones(2, 1, CV_64F);
    Mat testOutput = Mat::ones(1, 1, CV_64F);
    vector<Mat> testInputs;
    testInputs.push_back(testInput);
    vector<Mat> testOutputs;
    testOutputs.push_back(testOutput);
    net.evaluate(testInputs, testOutputs);

//    Mat testInput = Mat::zeros(2, 1, CV_64F);
//    cout << endl << net.feedfoward(testInput) << endl;
//    vector<Mat> trainingImages;
//    vector<int8_t> trainingLabels;
//    readImages("../train-images.idx3-ubyte", trainingImages);
//    readLabels("../train-labels.idx1-ubyte", trainingLabels);
    return 0;
}
