// This api is used for object detection with mobilenet+ssd models.
// 201707 wujx

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <setjmp.h>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
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
using tensorflow::uint8;

using namespace std;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    return tensorflow::errors::NotFound("Labels file ", file_name,
                                        " not found.");
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = data.ToString();
  return Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;
  tensorflow::Output image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  // auto float_caster =
  //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

  auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

  // Bilinearly resize the image to fit the required dimensions.
  // auto resized = ResizeBilinear(
  //     root, dims_expander,
  //     Const(root.WithOpName("size"), {input_height, input_width}));


  // Subtract the mean and divide by the scale.
  // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
  //     {input_std});


  //cast to int
  //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}


// Analyzes the output of the MultiBox graph to retrieve the highest scores and
// their positions in the tensor, which correspond to individual box detections.
Status GetTopDetections(const std::vector<Tensor>& outputs, int how_many_labels,
                        Tensor* indices, Tensor* scores) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}


// Converts an encoded location to an actual box placement with the provided
// box priors.
void DecodeLocation(const float* encoded_location, const float* box_priors,
                    float* decoded_location) {
  bool non_zero = false;
  for (int i = 0; i < 4; ++i) {
    const float curr_encoding = encoded_location[i];
    non_zero = non_zero || curr_encoding != 0.0f;

    const float mean = box_priors[i * 2];
    const float std_dev = box_priors[i * 2 + 1];

    float currentLocation = curr_encoding * std_dev + mean;

    currentLocation = std::max(currentLocation, 0.0f);
    currentLocation = std::min(currentLocation, 1.0f);
    decoded_location[i] = currentLocation;
  }

  if (!non_zero) {
    LOG(WARNING) << "No non-zero encodings; check log for inference errors.";
  }
}


float DecodeScore(float encoded_score) { return 1 / (1 + exp(-encoded_score)); }


void DrawBox(const int image_width, const int image_height, int left, int top,
             int right, int bottom, tensorflow::TTypes<uint8>::Flat* image) {
  tensorflow::TTypes<uint8>::Flat image_ref = *image;

  top = std::max(0, std::min(image_height - 1, top));
  bottom = std::max(0, std::min(image_height - 1, bottom));

  left = std::max(0, std::min(image_width - 1, left));
  right = std::max(0, std::min(image_width - 1, right));

  for (int i = 0; i < 3; ++i) {
    uint8 val = i == 2 ? 255 : 0;
    for (int x = left; x <= right; ++x) {
      image_ref((top * image_width + x) * 3 + i) = val;
      image_ref((bottom * image_width + x) * 3 + i) = val;
    }
    for (int y = top; y <= bottom; ++y) {
      image_ref((y * image_width + left) * 3 + i) = val;
      image_ref((y * image_width + right) * 3 + i) = val;
    }
  }
}


Status SaveImage(const Tensor& tensor, const string& file_path) {
  LOG(INFO) << "Saving image to " << file_path;
  CHECK(tensorflow::StringPiece(file_path).ends_with(".png"))
      << "Only saving of png files is supported.";

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string encoder_name = "encode";
  string output_name = "file_writer";

  tensorflow::Output image_encoder =
      EncodePng(root.WithOpName(encoder_name), tensor);
  tensorflow::ops::WriteFile file_saver = tensorflow::ops::WriteFile(
      root.WithOpName(output_name), file_path, image_encoder);

  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));

  std::vector<Tensor> outputs;
  TF_RETURN_IF_ERROR(session->Run({}, {}, {output_name}, &outputs));

  return Status::OK();
}



int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string image(argv[1]);
  string graph ="/data/wjx/workspace/mydata/my_frozen_graph/mobilenet/output_inference_graph_2class_26108.pb";
  string labels ="/data/wjx/workspace/mydata/my_frozen_graph/mobilenet/pascal_label_map.pbtxt";
  string image_out = "/data/wjx/workspace/git/tensorflow_pro/data/image_out.png";
  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "image_tensor:0";
  vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };
  //int top_classes = 5;
  float thres_hold = 0.6;
  string root_dir = "";

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  //LOG(ERROR) << "graph_path:" << graph_path;
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "LoadGraph ERROR!!!!"<< load_graph_status;
    return -1;
  }
  else printf("------- Load graph done.\n");

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> image_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir, image);
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                              input_std, &image_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  else printf("------- Read image tensor done.\n");


  cout << "======" << image_tensors.size() << endl;
  for(int i=0; i<image_tensors.size(); i++)
    cout << "======" << image_tensors[i].shape().DebugString() << endl;


  const Tensor& resized_tensor = image_tensors[0];
  cout <<"--resized shape: " << resized_tensor.shape().DebugString()<< ", tensors len: " 
       << image_tensors.size() << ", tensor type:"<< resized_tensor.dtype() << endl;
  
  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   output_layer, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  else cout << "------- Run model done." << endl;


  //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
  tensorflow::TTypes<float>::Flat Scores = outputs[1].flat<float>();
  tensorflow::TTypes<float>::Flat Classes = outputs[2].flat<float>();
  tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
  auto boxes = outputs[0].flat_outer_dims<float,3>();

  cout << "--num_detections:" << num_detections(0) << ", shape: " << outputs[0].shape().DebugString() << endl;


  int count_ = 0;
  vector<int> indices_;
  vector<float> scores_;
  for(size_t i = 0; i < num_detections(0) && i < 20; ++i)
  {
    if(Scores(i) > thres_hold) //
    {
      count_++;
      indices_.push_back(i);
      scores_.push_back(Scores(i));
      cout << "--index: " << i << " --- score:" << Scores(i) << " --- class:" << Classes(i)<< " --- box: " 
           << boxes(0,i,0) << ", " << boxes(0,i,1) << ", " << boxes(0,i,2)<< ", " << boxes(0,i,3) << endl;
    }
  }

//-------------------------------------------------------------------------------
  //Tensor indices;
  //Tensor scores;
  //int how_many_labels = 5;
  
  //GetTopDetections(outputs, how_many_labels, &indices, &scores);  
  //printf("------000\n");  

  //tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  //printf("------01010\n");
  //tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>(); // problem ???
  //tensorflow::TTypes<float>::Flat indices_flat = indices.flat<float>();
  //printf("------111\n");
  
  //const Tensor& encoded_locations = outputs[0];
  //auto locations_encoded = encoded_locations.flat<float>();

  Tensor *original_tensor = &image_tensors[0];
  LOG(INFO) << original_tensor->shape().DebugString();
 
  const int image_width = original_tensor->shape().dim_size(2);
  const int image_height = original_tensor->shape().dim_size(1);
  printf("-----image_width=%d------image_height=%d\n", image_width, image_height); 
 
  tensorflow::TTypes<uint8>::Flat image_flat = original_tensor->flat<uint8>();
//----------------------------------------------------------------- 
  cv::Mat mat = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  //float scale_factor_w = (float)input_width/mat.cols;
  //float scale_factor_h = (float)input_height/mat.rows;
  //printf("---cols = %d,  ---rows = %d\n", mat.cols, mat.rows);
  //printf("---scale_factor_w = %f,  ---scale_factor_h = %f\n", scale_factor_w, scale_factor_h);
//----------------------------------------------------------------
 
  cout << "===== Top " << count_ << " Detections Coord======" << endl;
  for (int pos = 0; pos < count_; ++pos) {
    int label_index = indices_[pos];
    float score = scores_[pos]; 
    
    float top =  boxes(0,label_index,0)* image_height;
    float left =  boxes(0,label_index,1)* image_width;
    float bottom =  boxes(0,label_index,2)* image_height;
    float right =  boxes(0,label_index,3)* image_width;
 
    //top = top/scale_factor_h;
    //left= left/scale_factor_w;
    //bottom= bottom/scale_factor_h;
    //right= right/scale_factor_w;
 
    cout      << "Index " << pos << ": "
              << "L:" << left << " "
              << "T:" << top << " "
              << "R:" << right << " "
              << "B:" << bottom << " "
              << "(" << label_index << ") score: " << score << endl;
    
    cv::rectangle(mat, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0), 2);
    char text[64];
    sprintf(text, "%lf", Scores(label_index));
    cv::putText(mat, text
                    , cv::Point(left, top)
                    , CV_FONT_HERSHEY_COMPLEX
                    , 0.8
                    , cv::Scalar(0, 0, 255));

    //DrawBox(image_width, image_height, left, top, right, bottom, &image_flat);
  }
  cv::imwrite(image_out, mat);   

  //if (!image_out.empty()) {
  //  SaveImage(*original_tensor, image_out);
  //}

  return 0;
}




