#include "nn.h"
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;
using namespace std;

#define EXTENDED_TRACE 1

NN::NN(std::vector<int>& config) :
    layers(config) {
    weights.reserve(config.size() - 1);
    biases.reserve(config.size() - 1);
    for(int i = 1; i < config.size(); ++i) {
        Mat weight(config.at(i), config.at(i - 1), CV_64F);
        randn(weight, 0, 1);
        Mat bias(config.at(i), 1, CV_64F);
        randn(bias, 0, 1);
        weights.push_back(weight);
        biases.push_back(bias);
    }
}

void NN::traceConfig() {
    cout << "NN configuration:" << endl
         << "  Layers: " << layers.size() << endl
         << "  Neurons: ";
    for (int i = 0; i < layers.size(); ++i) {
        cout << layers.at(i) << " ";
    }
    cout << endl << "  Weights: " << "(" << weights.size() <<")" << endl;
    utils::trace(weights);
    cout << endl << "  biases: " << "(" << biases.size() << ")" << endl;
    utils::trace(biases);
}

void NN::train(std::vector<Mat>& input,
               std::vector<Mat>& desiredOutput,
               int epochItemCount,
               int epochCount,
               double learningRate) {
    if (input.empty() ||
            input.size() != desiredOutput.size() ||
            epochCount == 0 ||
            epochCount == 0 ||
            !validate(input.at(0))) {
        return;
    }
    std::vector<int> indexes(input.size());
    for (int i = 0; i < epochCount; ++i) {
        random_shuffle(indexes.begin(), indexes.end());
        std::vector<int> epochIndexes(indexes.begin(), indexes.begin() + epochItemCount);
        trainInternal(input, desiredOutput, epochIndexes, learningRate);
    }
}

void NN::trainInternal(std::vector<Mat> &data,
                       std::vector<Mat> &desiredOutput,
                       std::vector<int> &indexes,
                       double learningRate) {
#if EXTENDED_TRACE
    cout << "RUN TRAINING WITH " << indexes.size() << " SAMPLES:" << endl;
#endif
    std::vector<Mat> sumWeightDerivative;
    std::vector<Mat> sumBiasDerivative;
    //init temporary storage to accamulate sum of each layer weights changes across all provided
    //to this method call samples
    for (int i = 0; i < weights.size(); i++) {
        sumWeightDerivative.push_back(Mat::zeros(weights[i].rows, weights[i].cols, CV_64F));
        sumBiasDerivative.push_back(Mat::zeros(biases[i].rows, biases[i].cols, CV_64F));
    }
    for (int i = 0; i < indexes.size(); ++i) {
        int index = indexes[i];
        Mat input = data[i];
        Mat output = desiredOutput[i];
        std::vector<Mat> weightDerivative;
        std::vector<Mat> biasDerivative;
        //backpropagate each input and output to calculate weights and biases change for this
        //particular sample
        backpropagate(input, output, weightDerivative, biasDerivative);
        for (int i = 0; i < weights.size(); i++) {
            sumWeightDerivative[i] += weightDerivative[i];
            sumBiasDerivative[i] += biasDerivative[i];
        }
    }

#if EXTENDED_TRACE
    cout << "   WEIGHTS BEFORE UPDATE:" << endl;
    utils::trace(weights);
    cout << "   BIASES BEFORE UPDATE:" << endl;
    utils::trace(biases);
#endif

    //compute average weights and biases changes
    double multiplier = learningRate / indexes.size();
    for (int i = 0; i < weights.size(); i++) {
        sumWeightDerivative[i] = sumWeightDerivative[i] * multiplier;
        sumBiasDerivative[i] = sumBiasDerivative[i] * multiplier;
        //update global weights and biases
        weights[i] -= sumWeightDerivative[i];
        biases[i] -= sumBiasDerivative[i];
    }

#if EXTENDED_TRACE
    cout << "   WEIGHTS AFTER UPDATE:" << endl;
    utils::trace(weights);
    cout << "   BIASES AFTER UPDATE:" << endl;
    utils::trace(biases);
#endif
}

void NN::backpropagate(Mat &input,
                       Mat &desiredOutput,
                       std::vector<Mat> &weightDerivative,
                       std::vector<Mat> &biasDerivative) {
#if EXTENDED_TRACE
    cout << "BACKPROPAGATE:" << endl
         << "INPUT:" << input << endl
         << "OUTPUT:" << desiredOutput << endl;
#endif

    //create space for storing partial derivatives in respect to weights and biases from each layer
    weightDerivative.resize(weights.size());
    biasDerivative.resize(biases.size());

#if EXTENDED_TRACE
    cout << "BACKPROP FORWARD:" << endl;
#endif

    //feedforward input example and save activations on each layer
    //save each layer activations after applying sigmoid activation function
    std::vector<Mat> activations;
    activations.reserve(layers.size());
    //save each layer results before applying sigmoid activation function
    std::vector<Mat> results;
    results.reserve(layers.size() - 1);
    activations.push_back(input);

    //iterate through layers and compute activations
    Mat feed = input.clone();
    for (int i = 0; i < weights.size(); ++i) {
        feed = weights.at(i) * feed + biases.at(i);
        results.push_back(feed.clone());
        feed = utils::sigmoid(feed);
        activations.push_back(feed.clone());
    }

#if EXTENDED_TRACE
    cout << "ACTIVATIONS:" << endl;
    utils::trace(activations);
    cout << "RESULTS:" << endl;
    utils::trace(results);

    cout << "BACKPROP CALCULATE OUTPUT LAYER:" << endl;
#endif

    //copute the last layer errors before backpropagation start
    Mat delta;
    multiply(
             utils::costFunctionDerivative(activations.back(), desiredOutput),
             utils::sigmoidDerivative(results.back()),
             delta
    );

#if EXTENDED_TRACE
    cout << "THE LAST LAYER DELTA:" << delta << endl;
#endif

    //use last error for bias change
    biasDerivative[biasDerivative.size() - 1] = delta;
    //get activations from the previous layer to compute the current layer weights change
    Mat prevActivation = (*(activations.end() - 2));

#if EXTENDED_TRACE
    cout << "PREVIOUS LAYER ACTIVATION:" << endl << prevActivation << endl;
#endif

    //compute the last layer weights change
    Mat lastWeightDervitive = delta * prevActivation.t();
    weightDerivative[weightDerivative.size() - 1] = lastWeightDervitive;

#if EXTENDED_TRACE
    cout << "DERVIVATIVES BEFORE LOOP START:" << endl;
    utils::trace(weightDerivative);
    cout << endl;
    utils::trace(biasDerivative);
#endif

    //iterate through the rest of the layers and propagate error to compute partial derivatives
    for (int i = weights.size() - 2; i >= 0; --i) {
#if EXTENDED_TRACE
        cout << "_______BACKPROPAGATION LOOP ITERATION:" << i << "_______" << endl;
#endif
        //calculate sigmoid function derivative for the current layer results
        Mat result = results.at(i);
        Mat activation = utils::sigmoidDerivative(result);

#if EXTENDED_TRACE
        cout << "   RESULT:" << endl << result << endl;
        cout << "   ACTIVATION:" << endl << activation << endl;
        cout << "   DELTA:" << endl << delta << endl;
        cout << "   WEIGTHS:" << endl << weights.at(i + 1).t() << endl;
#endif

        //get weights from the previous level and propagate error
        Mat weightsWithDelta = weights.at(i + 1).t() * delta;

        //finally calculate current layers errors
        Mat newDelta;
        multiply(weightsWithDelta, activation, newDelta);
        //save current layers error
        delta = newDelta;

#if EXTENDED_TRACE
        cout << "   DELTA:" << delta << endl;
#endif

        //use errors to change current level biases
        biasDerivative[i] = delta;
        //get previous activations to compute weights change
        prevActivation = activations[i];

#if EXTENDED_TRACE
        cout << "   PREVIOUS ACTIVATION:" << endl << prevActivation << endl;
#endif

        //compute current layer weights change
        weightDerivative[i] = delta * prevActivation.t();


#if EXTENDED_TRACE
        cout << "____________________________________________"<< endl;
#endif
    }
}

int NN::evaluate(std::vector<Mat>& input, std::vector<Mat>& desiredOutput) {
//    Mat initial = feedfoward(input.at(0));
//    cout << "INITIAL RESULT:" <<  endl << initial << endl;
//    cout << "UPDATE:" << endl;
//    vector<Mat> wd;
//    vector<Mat> ad;
//    backpropagate(input.at(0), desiredOutput.at(0), wd, ad);
//    cout << "WD: " << endl;
//    utils::trace(wd);
//    cout << "AD: " << endl;
//    utils::trace(ad);
//    for (int i = 0; i < weights.size(); ++i) {
//        cout << "ITERATION: " << i;
//        weights.insert(weights.begin() + i, weights.at(i) - wd.at(i));
//        biases.insert(biases.begin() + i, biases.at(i) - ad.at(i));
//    }
//    traceConfig();
//    Mat result = feedfoward(input.at(0));
//    cout << "AFTER RESULT:" <<  endl << result << endl;
#if EXTENDED_TRACE
    cout << "EVALUATE: " << input.size() << " SAMPLES" << endl;
#endif
    assert(input.size() == desiredOutput.size());
    for (int i = 0; i < input.size(); ++i) {
        Mat computedOutput = feedfoward(input[i]);
        assert(computedOutput.cols == 1);
        assert(computedOutput.rows == layers.back());
        assert(computedOutput.cols == desiredOutput[i].cols);
        assert(computedOutput.rows == desiredOutput[i].rows);
        double maxComputedOutput = computedOutput.at(0, 0);
        double maxComputedIndex = 0;
        for (int j = 1; j < computedOutput.rows; ++j) {
            if (computedOutput.at(0, i) > maxComputedOutput) {
                maxComputedOutput = computedOutput.at(0, i);
                maxComputedIndex = i;
            }
        }
    }

}

bool NN::validate(Mat &data) {
    if (data.cols * data.rows != layers.at(0)) {
        return false;
    }
    return true;
}

Mat NN::feedfoward(Mat &input) {
   Mat feed = input.clone();
   for (int i = 0; i < weights.size(); ++i) {
       feed = weights.at(i) * feed + biases.at(i);
       feed = utils::sigmoid(feed);
   }
   return feed;
}

int NN::getLayersCount() {
    return layers.size();
}