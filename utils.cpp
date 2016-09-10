#include "utils.h"

namespace utils {
    static inline double sigmoid(double input) {
        return 1.0 / (1.0 + exp((-1) * input));
    }

    cv::Mat sigmoid(const cv::Mat& input) {
        cv::Mat simoided = input.clone();
        for(int row = 0; row < simoided.rows; ++row) {
            for (int col = 0; col < simoided.cols; ++col) {
                simoided.at<double>(row, col) = sigmoid(simoided.at<double>(row, col));
            }
        }
        return simoided;
    }

    double sigmoidDerivative(const double input) {
        return sigmoid(input) * (1 - sigmoid(input));
    }

    cv::Mat sigmoidDerivative(const cv::Mat& input) {
        cv::Mat output;
        cv::multiply(sigmoid(input), (1 - sigmoid(input)), output);
        return output;
    }

    cv::Mat costFunctionDerivative(const cv::Mat& outputActivations,
                                          const cv::Mat& desiredOutput) {
        return outputActivations - desiredOutput;
    }
}
