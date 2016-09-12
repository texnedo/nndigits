#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>
#include <array>
#include <math.h>
#include "utils.h"
#include "nn.h"

using namespace std;
using namespace cv;

typedef vector<Mat> MAT_VEC;

#define SHOW_EVERY_IMAGE 0
#define SHOW_ALL_IMAGES 0
#define SHOW_VALIDATE_IMAGES 0

int readImages(const char* fileName, MAT_VEC& images) {
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
    Mat allImages(lineSize * numberOfRow, lineSize * numberOfColumns, CV_64F);
    int imageColumnOffset = numberOfColumns * lineSize;
#endif
    for (int i = 0; i < numberOfImages; ++i) {
        Mat image(numberOfRow, numberOfColumns, CV_64F);
        double* imagePoint = (double*) image.data;
        for (int j = 0; j < imageSize; ++j) {
            uchar data;
            trainingData.read((char*)&data, 1);
            double ddata = (double) data;
            (*imagePoint) = ddata;
            imagePoint++;
        }
        image.rows *= image.cols;
        image.cols = 1;
        images.push_back(image);
#if SHOW_ALL_IMAGES
        int filledLines = i / lineSize;
        //calculate offset in the big matrix
        double* allImagesPtr = (double*)allImages.data +
                (filledLines * imageSize) * lineSize + (i - filledLines * lineSize) * numberOfColumns;
        for (int j = 0; j < numberOfRow; ++j) {
            memcpy(allImagesPtr, image.row(j).data, numberOfColumns * sizeof(double));
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
    return images.size();
}

int readLabels(const char* fileName, MAT_VEC& lables) {
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
    vector<int8_t> data;
    data.resize(numberOfLables);
    lables.reserve(numberOfLables);
    trainingLables.read((char*)data.data(), numberOfLables);
    for (auto it = data.begin(); it != data.end(); ++it) {
        Mat output = Mat::zeros(10, 1, CV_64F);
        output.at<double>(*it, 0) = 1;
        lables.push_back(output);
    }
    cout << "Lablels count: " << lables.size() << endl;
    return lables.size();
}

void readMnistData(MAT_VEC& trainingImages,
                   MAT_VEC& trainingLabels,
                   MAT_VEC& validateImages,
                   MAT_VEC& validateLabels) {
    readImages("../train-images.idx3-ubyte", trainingImages);
    readLabels("../train-labels.idx1-ubyte", trainingLabels);
    readImages("../t10k-images.idx3-ubyte", validateImages);
    readLabels("../t10k-labels.idx1-ubyte", validateLabels);
    assert(trainingImages.size() == trainingLabels.size());
    assert(trainingImages.size() > 1);
    assert(validateImages.size() == validateLabels.size());
    assert(validateImages.size() > 1);
}

void showImage(Mat& imgae, Mat& label) {
    Mat imgaeToShow = imgae.clone();
    int size = sqrt(max(imgaeToShow.cols, imgaeToShow.rows));
    imgaeToShow.cols = size;
    imgaeToShow.rows = size;
    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", imgaeToShow);
    cout << "Expected label: "  << endl << label << endl;
    waitKey(0);
}

void showMnistData(MAT_VEC& images, MAT_VEC& labels) {
    for (int i = 0; i < images.size(); ++i) {
        showImage(images[i], labels[i]);
    }
}

int main(int argc, char *argv[]) {
//    vector<int> config = {2, 3, 2, 1};
//    NN net(config);
//    net.traceConfig();
//    Mat testInput = Mat::ones(2, 1, CV_64F);
//    Mat testOutput = Mat::ones(1, 1, CV_64F);
//    MAT_VEC testInputs;
//    testInputs.push_back(testInput);
//    MAT_VEC testOutputs;
//    testOutputs.push_back(testOutput);
//    net.evaluate(testInputs, testOutputs);

//    Mat testInput = Mat::zeros(2, 1, CV_64F);
//    cout << endl << net.feedfoward(testInput) << endl;

//    MAT_VEC trainingImages;
//    MAT_VEC trainingLabels;
//    readImages("../train-images.idx3-ubyte", trainingImages);
//    readLabels("../train-labels.idx1-ubyte", trainingLabels);
//    MAT_VEC tenkImages;
//    MAT_VEC tenkLabels;
//    readImages("../t10k-images.idx3-ubyte", tenkImages);
//    readLabels("../t10k-labels.idx1-ubyte", tenkLabels);
//    assert(trainingImages.size() == trainingLabels.size());
//    assert(trainingImages.size() > 1);


//    for (int i = 0; i < trainingImages.size(); ++i) {
//            trainingImages[i].cols = 28;
//            trainingImages[i].rows = 28;
//            namedWindow("test image", WINDOW_AUTOSIZE);
//            imshow("test image", trainingImages[i]);
//            cout << trainingLabels[i] << endl;
//            waitKey(0);
//    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    vector<int> config = { trainingImages[0].rows * trainingImages[0].cols, 30, trainingLabels[0].rows };
//    NN net(config);
//    net.train(trainingImages, trainingLabels, 30, 10, 10);
//    cout << net.evaluate(tenkImages, tenkLabels) << "/" << tenkLabels.size() << endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    vector<int> config = { trainingImages[0].rows * trainingImages[0].cols, 30, trainingLabels[0].rows };
//    NN net(config);
//    net.traceConfig();
//    net.train(trainingImages, trainingLabels, 30, 30, 0.01);
//    net.traceConfig();
//    cout << net.evaluate(tenkImages, tenkLabels) << "/" << tenkImages.size() << endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    vector<int> config = { trainingImages[0].rows * trainingImages[0].cols, 30, trainingLabels[0].rows };
//    NN net(config);
//    MAT_VEC input { trainingImages[0], trainingImages[1] };
//    MAT_VEC output { trainingLabels[0], trainingLabels[1] };
//    net.train(input, output, 2, 2, 1);
//    cout << net.evaluate(input, output) << "/" << input.size() << endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    vector<int> config = { trainingImages[0].rows * trainingImages[0].cols, 30, trainingLabels[0].rows };
//    NN net(config);
//    MAT_VEC input { trainingImages[0] };
//    MAT_VEC output { trainingLabels[0] };
//    net.train(input, output, 1, 1, 10);
//    cout << net.evaluate(input, output) << "/" << input.size() << endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    for (int i = 0; i < trainingImages.size(); ++i) {
//        vector<int> config = { trainingImages[0].rows * trainingImages[0].cols, 30, trainingLabels[0].rows };
//        NN net(config);
//        MAT_VEC input { trainingImages[i] };
//        MAT_VEC output { trainingLabels[i] };
//        net.train(input, output, 1, 1, 0.1);
//        cout << net.evaluate(input, output) << "/" << input.size() << endl;
//    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    for (int i = 0; i < trainingImages.size(); ++i) {
//        MAT_VEC input { trainingImages[i] };
//        MAT_VEC output { trainingLabels[i] };
//        cout << net.evaluate(input, output) << endl;

//        trainingImages[i].cols = 28;
//        trainingImages[i].rows = 28;
//        namedWindow("test image", WINDOW_AUTOSIZE);
//        imshow("test image", trainingImages[i]);
//        waitKey(0);
//    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//    vector<int> config = { 4, 10, 3 };
//    NN net(config);
//    net.traceConfig();
//    double inputData[] = {1, 2, 3, 4};
//    double inputData1[] = {6, 7, 3, 9};
//    double inputData2[] = {3, 4, 9, 1};
//    double outputData[] = {1, 0, 0};
//    double outputData1[] = {0, 1, 0};
//    double outputData2[] = {0, 0, 1};
//    MAT_VEC input { Mat(4, 1, CV_64F, inputData), Mat(4, 1, CV_64F, inputData1), Mat(4, 1, CV_64F, inputData2) };
//    MAT_VEC output { Mat(3, 1, CV_64F, outputData), Mat(3, 1, CV_64F, outputData1), Mat(3, 1, CV_64F, outputData2) };
//    net.train(input, output, 2, 1000, 1);
//    net.traceConfig();
//    cout << "result" << endl;
//    Mat result = net.feedfoward(*(input.begin() + 0));
//    cout << result;
//    cout << "evaluate:" << endl;
//    cout << net.evaluate(input, output) << "/" << input.size() << endl;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
    MAT_VEC trainingImages;
    MAT_VEC trainingLabels;
    MAT_VEC validateImages;
    MAT_VEC validateLabels;
    readMnistData(trainingImages, trainingLabels, validateImages, validateLabels);
//    showMnistData(trainingImages, trainingLabels);
    int inputSize = trainingImages[0].rows;
    int outputSize = trainingLabels[0].rows;
    vector<int> config = { inputSize, 30, outputSize };
    NN net(config);
    cout << trainingImages.size() << " " << trainingLabels.size() << endl;
    net.train(trainingImages, trainingLabels, 1000, 20000, 5);

    cout << "evaluate:" << endl;
    cout << net.evaluate(trainingImages, trainingLabels) << "/" << trainingLabels.size() << endl;

    cout << "validate:" << endl;
    cout << net.evaluate(validateImages, validateLabels) << "/" << validateLabels.size() << endl;

#if SHOW_VALIDATE_IMAGES
    for (int i = 0; i < validateImages.size(); ++i) {
        showImage(validateImages[i], validateLabels[i]);
        Mat result = net.feedfoward(validateImages[i]);
        cout << "Computed: "  << endl << result << endl;
    }
#endif

    return 0;
}
