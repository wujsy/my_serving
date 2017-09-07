/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// A minimal but useful C++ example showing how to load an Imagenet-style object
// recognition TensorFlow model, prepare input images for it, run them through
// the graph, and interpret the results.
//
// It's designed to have as few dependencies and be as clear as possible, so
// it's more verbose than it could be in production code. In particular, using
// auto for the types of a lot of the returned values from TensorFlow calls can
// remove a lot of boilerplate, but I find the explicit types useful in sample
// code to make it simple to look up the classes involved.
//
// To use it, compile and then run in a working directory with the
// learning/brain/tutorials/label_image/data/ folder below it, and you should
// see the top five labels for the example Lena image output. You can then
// customize it to use your own models or images by changing the file names at
// the top of the main() function.
//
// The googlenet_graph.pb file included by default is created from Inception.
//
// Note that, for GIF inputs, to reuse existing code, only single-frame ones
// are supported.

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/padding_fifo_queue.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "cv_process.hpp" 

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
    std::unique_ptr<tensorflow::Session>* session, tensorflow::GraphDef& graph_def) {
    //  tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
            graph_file_name, "'");
    }

    WriteTextProto(tensorflow::Env::Default(), "/tmp/test_inception_v4.pbtxt", graph_def);

    tensorflow::SessionOptions options_;
    options_.config.set_allow_soft_placement(true);

    session->reset(tensorflow::NewSession(options_));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

struct c_tf_detect_result {
    int nclass;
    float score;
    cv::Rect r;
};

void prepare_tf_detect_result(tensorflow::Tensor& boxes, tensorflow::Tensor& classes, tensorflow::Tensor& scores, 
    float scale_factor_w, float scale_factor_h, float threshold,
    std::map<int, std::vector<c_tf_detect_result> > & dst) {
    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
    tensorflow::TTypes<long long>::Flat classes_flat = classes.flat<long long>();
    typename TTypes<float, 3>::Tensor obox = boxes.tensor<float, 3>();
    dst.clear();
    for (int i = 0; i < scores_flat.size(); ++i) {
        if (scores_flat(i) >= threshold) {
            c_tf_detect_result nr;
            nr.nclass = classes_flat(i);
            nr.score = scores_flat(i);
            int cx = obox(0, i, 0);
            int cy = obox(0, i, 1);
            int w = obox(0, i, 2);
            int h = obox(0, i, 3);
            int x1 = cx - w / 2;
            int y1 = cy - h / 2;
            int x2 = cx + w / 2;
            int y2 = cy + h / 2;
            x1 /= scale_factor_w;
            x2 /= scale_factor_w;
            y1 /= scale_factor_h;
            y2 /= scale_factor_h;
            nr.r = cv::Rect(x1 < x2 ? x1 : x2, y1 < y2 ? y1 : y2, abs(x1 - x2), abs(y1 - y2));
            
            auto iter = dst.find(nr.nclass);
            if (iter == dst.end()) {
                std::vector<c_tf_detect_result> s;
                s.push_back(nr);
                dst.insert(std::make_pair(nr.nclass, s));
            }
            else {
                iter->second.push_back(nr);
            }
        }
    }
}

void nms_plus(std::map<int, std::vector<c_tf_detect_result> >& input, std::map<int, std::vector<c_tf_detect_result> >& output, float thresh) {
    output.clear();

    for (auto iter = input.begin(); iter != input.end(); ++iter) {
        std::vector<c_tf_detect_result>& src = iter->second;
        std::sort(src.begin(), src.end(), [](c_tf_detect_result c1, c_tf_detect_result c2) { return c1.score < c2.score; });

        std::vector<c_tf_detect_result> dst;
        while (src.size() > 0)
        {
            // grab the last rectangle
            auto lastElem = --std::end(src);
            const cv::Rect& rect1 = lastElem->r;

            src.erase(lastElem);

            for (auto pos = std::begin(src); pos != std::end(src); )
            {
                // grab the current rectangle
                const cv::Rect& rect2 = pos->r;

                float intArea = (rect1 & rect2).area();
                float unionArea = rect1.area() + rect2.area() - intArea;
                float overlap = intArea / unionArea;

                // if there is sufficient overlap, suppress the current bounding box
                if (overlap >= thresh)
                {
                    pos = src.erase(pos);
                }
                else
                {
                    ++pos;
                }
            }

            dst.push_back(*lastElem);
        }

        output.insert(std::make_pair(iter->first, dst));
    }
}

int main(int argc, char* argv[]) {
    // These are the command-line flags the program can understand.
    // They define where the graph and input data is located, and what kind of
    // input the model expects. If you train your own model, or use something
    // other than inception_v3, then you'll need to update these.
    string image = "tensorflow/examples/label_image/data/grace_hopper.jpg";
    string out_image = "./test.png";
    string ckpt = "";
    string graph =
        "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb";
    string labels =
        "tensorflow/examples/label_image/data/imagenet_slim_labels.txt";
    int32 input_width = 299;
    int32 input_height = 299;
    int32 input_mean = 0;
    int32 input_std = 255;
    float thres_hold = 0.6;
    string input_layer = "image_input";
    string output_layer = "InceptionV3/Predictions/Reshape_1";
    bool self_test = false;
    string root_dir = "";
    std::vector<Flag> flag_list = {
        Flag("image", &image, "image to be processed"),
        Flag("out_image", &out_image, "output image"),
        Flag("graph", &graph, "graph to be executed"),
        Flag("thres_hold", &thres_hold, "probability thres_hold"),
        Flag("ckpt", &ckpt, "check point"),
        Flag("labels", &labels, "name of file containing labels"),
        Flag("input_width", &input_width, "resize image to this width in pixels"),
        Flag("input_height", &input_height,
                "resize image to this height in pixels"),
        Flag("input_mean", &input_mean, "scale pixel values to this mean"),
        Flag("input_std", &input_std, "scale pixel values to this std deviation"),
        Flag("input_layer", &input_layer, "name of input layer"),
        Flag("output_layer", &output_layer, "name of output layer"),
        Flag("self_test", &self_test, "run a self test"),
        Flag("root_dir", &root_dir,
                "interpret image and graph file names relative to this directory"),
    };
    string usage = tensorflow::Flags::Usage(argv[0], flag_list);
    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) {
        LOG(ERROR) << usage;
        return -1;
    }

    // We need to call this to set up global state for TensorFlow.
    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }

    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    tensorflow::GraphDef graph_def;

    Status load_graph_status = LoadGraph(graph_path, &session, graph_def);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }
// Get the image from disk as a float array of numbers, resized and normalized
// to the specifications the main graph expects.
    std::vector<Tensor> resized_tensors;
    string image_path = tensorflow::io::JoinPath(root_dir, image);

    cv::Mat mat = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
    std::vector<unsigned char> odata;
    cv::Mat img;
    cvprocess::resizeImage(mat, 103.939, 116.779, 123.68, input_width, input_height, img);

    Tensor inputImg(tensorflow::DT_FLOAT, { 1,input_height,input_width,3 });
    auto inputImageMapped = inputImg.tensor<float, 4>();
    //Copy all the data over
    for (int y = 0; y < input_height; ++y) {
        const float* source_row = ((float*)img.data) + (y * input_width * 3);
        for (int x = 0; x < input_width; ++x) {
            const float* source_pixel = source_row + (x * 3);
            inputImageMapped(0, y, x, 0) = source_pixel[0];
            inputImageMapped(0, y, x, 1) = source_pixel[1];
            inputImageMapped(0, y, x, 2) = source_pixel[2];
        }
    }

    resized_tensors.push_back(inputImg);

    const Tensor& resized_tensor = resized_tensors[0];

    std::vector<Tensor> outputs;
    std::vector<string> olabels = { "bbox/trimming/bbox","probability/class_idx","probability/score" };

    tensorflow::TensorShape image_input_shape;
    image_input_shape.AddDim(1);
    image_input_shape.AddDim(384);
    image_input_shape.AddDim(624);
    image_input_shape.AddDim(3);

    tensorflow::TensorShape box_mask_shape;
    box_mask_shape.AddDim(1);
    box_mask_shape.AddDim(16848);
    box_mask_shape.AddDim(1);
    Tensor box_mask_tensor(tensorflow::DataType::DT_FLOAT, box_mask_shape);

    tensorflow::TensorShape box_delta_input_shape;
    box_delta_input_shape.AddDim(1);
    box_delta_input_shape.AddDim(16848);
    box_delta_input_shape.AddDim(4);
    Tensor box_delta_input_tensor(tensorflow::DataType::DT_FLOAT, box_delta_input_shape);

    tensorflow::TensorShape box_input_shape;
    box_input_shape.AddDim(1);
    box_input_shape.AddDim(16848);
    box_input_shape.AddDim(4);
    Tensor box_input_tensor(tensorflow::DataType::DT_FLOAT, box_input_shape);

    tensorflow::TensorShape labels_shape;
    labels_shape.AddDim(1);
    labels_shape.AddDim(16848);
    labels_shape.AddDim(3);
    Tensor labels_tensor(tensorflow::DataType::DT_FLOAT, labels_shape);

    std::cout << "-------------step 1 -------------" << std::endl;
    Status run_status = session->Run({ {"image_input", resized_tensor}, {"box_mask", box_mask_tensor}, {"box_delta_input", box_delta_input_tensor}, {"box_input", box_input_tensor}, {"labels", labels_tensor} },
        {}, { "fifo_queue_EnqueueMany" }, nullptr);

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    else {
        LOG(ERROR) << "Running model success: " << run_status;
    }

    std::cout << "-------------step 2 -------------" << std::endl;
    run_status = session->Run({},
    {}, { "batch/fifo_queue_enqueue" }, nullptr);

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    else {
        LOG(ERROR) << "Running model success: " << run_status;
    }

    std::cout << "------------step 3--------------" << std::endl;
    run_status = session->Run({},
        olabels, {}, &outputs);

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }
    else {
        LOG(ERROR) << "Running model success: " << run_status;

    }

    Tensor boxes = outputs[0];
    Tensor oindices = outputs[1];
    Tensor scores = outputs[2];

    float scale_factor_w = (float)img.cols / mat.cols;
    float scale_factor_h = (float)img.rows / mat.rows;

        std::map<int, std::vector<c_tf_detect_result> >  src;
        std::map<int, std::vector<c_tf_detect_result> >  dst;
        prepare_tf_detect_result(boxes, oindices, scores, scale_factor_w, scale_factor_h, thres_hold, src);
        nms_plus(src, dst, 0.4);

        for (auto it = dst.begin(); it != dst.end(); ++it) {
            std::vector<c_tf_detect_result>& cur = it->second;
            for (auto iter = cur.begin(); iter != cur.end(); ++iter) {
                int x1 = iter->r.x;
                int y1 = iter->r.y;
                int x2 = x1 + iter->r.width;
                int y2 = y1 + iter->r.height;
                cv::rectangle(mat, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
                char text[64];
                sprintf(text, "%lf", iter->score);
                cv::putText(mat, text
                    , cv::Point(x1, y1)
                    , CV_FONT_HERSHEY_COMPLEX
                    , 0.8
                    , cv::Scalar(0, 0, 255));
            }
        }
    cv::imwrite(out_image, mat);

    return 0;
}
