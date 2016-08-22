#ifndef UTILS_H
#define UTILS_H
#include "math.h"
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
namespace utils {
    inline double sigmoid(double input) {
        return 1.0 / (1.0 + exp((-1) * input));
    }

    inline cv::Mat sigmoid(const cv::Mat& input) {
        cv::Mat simoided = input.clone();
        for(int row = 0; row < simoided.rows; ++row) {
            for (int col = 0; col < simoided.cols; ++col) {
                simoided.at<double>(row, col) = sigmoid(simoided.at<double>(row, col));
            }
        }
        return simoided;
    }

    inline double sigmoidDerivative(const double input) {
        return sigmoid(input) * (1 - sigmoid(input));
    }

    inline cv::Mat sigmoidDerivative(const cv::Mat& input) {
        cv::Mat output;
        cv::multiply(sigmoid(input), (1 - sigmoid(input)), output);
        return output;
    }

    inline cv::Mat costFunctionDerivative(const cv::Mat& outputActivations,
                                          const cv::Mat& desiredOutput) {
        return outputActivations - desiredOutput;
    }

    inline void trace(const std::vector<cv::Mat>& data) {
        for (int i = 0; i < data.size(); ++i) {
            std::cout << data.at(i) << std::endl;
        }
    }
}
#endif // UTILS_H
