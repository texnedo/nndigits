#ifndef NN_H
#define NN_H
#include <opencv2/core/core.hpp>
#include <vector>

class NN {
public:
    NN(std::vector<int>& config);
    cv::Mat feedfoward(cv::Mat& input);
    int getLayersCount();
    void traceConfig();
    void train(std::vector<cv::Mat>& input,
            std::vector<cv::Mat>& desiredOutput,
            int epochItemCount,
            int epochCount,
            double learningRate
    );
    int evaluate(
            std::vector<cv::Mat>& input,
            std::vector<cv::Mat>& desiredOutput
    );

private:
     const std::vector<int> layers;
     std::vector<cv::Mat> weights;
     std::vector<cv::Mat> biases;
     bool validate(cv::Mat& data);
     void trainInternal(std::vector<cv::Mat> &data,
                        std::vector<cv::Mat> &desiredOutput,
                        std::vector<int> &indexes,
                        double learningRate
                        );
     void backpropagate(cv::Mat& input,
                        cv::Mat& desiredOutput,
                        std::vector<cv::Mat>& weightDerivative,
                        std::vector<cv::Mat>& biasDerivative
                        );
};

#endif // NN_H
