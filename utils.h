#ifndef UTILS_H
#define UTILS_H
#include "math.h"
#include <vector>
#include <stack>
#include <iostream>
#include <opencv2/core/core.hpp>
namespace utils {
    cv::Mat sigmoid(const cv::Mat& input);

    double sigmoidDerivative(const double input);

    cv::Mat sigmoidDerivative(const cv::Mat& input);

    cv::Mat costFunctionDerivative(const cv::Mat& outputActivations,
                                          const cv::Mat& desiredOutput);
    template <class T>
    void trace(const std::vector<T>& data) {
        for (int i = 0; i < data.size(); ++i) {
            std::cout << data.at(i) << " ";
        }
        std::cout << std::endl;
    }

    template <class T>
    void shuffle(const std::vector<T>& data, std::vector<T>& output) {
        assert(data.size() >= output.capacity());
        assert(data.size() != 0);
        std::vector<T> dataCopy(data);
        int maxDataIndex = dataCopy.size() - 1;
        for (int i = 0; i < output.capacity(); ++i) {
            int index = rand() % (maxDataIndex - i);
            T tmp = dataCopy[index];
            dataCopy[index] = dataCopy[maxDataIndex - i];
            dataCopy[maxDataIndex - i] = tmp;
            output.push_back(tmp);
        }
    }

    template <class T>
    void shuffleOptimal(std::vector<T>& data, std::vector<T>& output) {
        assert(data.size() >= output.capacity());
        assert(data.size() != 0);
        int maxDataIndex = data.size() - 1;
        std::stack<int> indexes;
        for (int i = 0; i < output.capacity(); ++i) {
            int index = rand() % (maxDataIndex - i);
            indexes.push(index);
            T tmp = data[index];
            data[index] = data[maxDataIndex - i];
            data[maxDataIndex - i] = tmp;
            output.push_back(tmp);
        }
        //restore back original order
        while(!indexes.empty()) {
            int position = indexes.size() - 1;
            int index = indexes.top();
            indexes.pop();
            T tmp = data[maxDataIndex - position];
            data[maxDataIndex - position] = data[index];
            data[index] = tmp;
        }
    }

}
#endif // UTILS_H
