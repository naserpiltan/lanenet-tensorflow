#include <QCoreApplication>
#include <tensorflow/cc/ops/array_ops.h>
#include "tensorflow/cc/client/client_session.h"

#include "tensorflow/cc/ops/standard_ops.h"

// tensorflow::Tensor
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/lib/core/status.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/protobuf/config.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/util/command_line_flags.h>
#include <opencv2/opencv.hpp>

#include <iostream>

using tensorflow::Scope;
using tensorflow::Output;
using tensorflow::Tensor;
using tensorflow::ClientSession;

using tensorflow::ops::Const;
using tensorflow::ops::MatMul;


using namespace std;
using namespace tensorflow;
using namespace cv;


int main(int argc, char *argv[])
{

    std::string PathGraph = "/media/eagle-soft/ExtDrive/lanedetection/Lanenet/lanenet/lanenet-lane-detection-master/lanenet.pb";

    int input_width_ = 512;                                                         /// Width of the input placeholder
    int input_height_ = 256;
    //initial declaration Tensorflow
    tensorflow::Session* session;
    tensorflow::Status status;
    status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    // Define Graph
    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);

    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    Tensor inputImg(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 256, 512, 3}));
    float *image1_p_ = inputImg.flat<float>().data();
    float * image2_p_ = image1_p_ + input_width_ * input_height_ * 3;
    Mat cv_image1_ = cv::Mat(input_height_, input_width_, CV_32FC3, image1_p_);
    Mat cv_image2_ = cv::Mat(input_height_, input_width_, CV_32FC3, image2_p_);


    Mat image;cv::Mat resized_img, resized_flipped_img;
    image=imread("/media/eagle-soft/ExtDrive/datasets/kitti/kitti-jpg/images/umm_000033.jpg");
    resize(image,resized_img,Size(512,256));
    //cvtColor(image,image,COLOR_BGR2RGB);
    //image.convertTo(image, CV_32FC3);
    //image = image / 127.5 - 1.0;
    resized_img.convertTo(cv_image1_, CV_32FC3, 1/127.0);
    resized_img = resized_img-1.0;




    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        { "lanenet/input_tensor", inputImg},
    };



    std::vector<tensorflow::Tensor> outputs;
    status = session->Run(inputs, {"lanenet/final_binary_output"},{}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    //tensorflow::Tensor output = std::move(outputs.at(0));
    Mat mask1_ = cv::Mat(256, 512, CV_32FC1);
    Mat mask2_ = cv::Mat(256, 512, CV_32FC1);

    float* depth1_p = outputs[0].flat<float>().data();

    float* depth1_flipped_p = depth1_p + 512 * 256;
    cv::Mat cv_depth1 = cv::Mat(input_height_, input_width_, CV_32FC1, depth1_p);
    normalize(cv_depth1,cv_depth1,0,255,cv::NORM_MINMAX);
    cv_depth1.convertTo(cv_depth1,CV_8UC1);
    //   cv::Mat cv_depth1_flipped = cv::Mat(input_height_, input_width_, CV_32FC1, depth1_flipped_p);
    //    cv::Mat cv_depth2;
    //    cv::flip(cv_depth1_flipped, cv_depth2, 1);
    //    Mat cv_depth_;
    //    cv_depth_ = 0.5f * (cv_depth1 + cv_depth2);
    //    cv_depth_ = cv_depth_.mul(1.0f - mask1_ - mask2_) + mask2_.mul(cv_depth1) + mask1_.mul(cv_depth2);


    //std::cerr << "final output size=" << output.shape() << std::endl;
    imshow("sss",cv_depth1);
    waitKey(0);
    cout<<"salam merci " <<endl;

    return 0;
}
