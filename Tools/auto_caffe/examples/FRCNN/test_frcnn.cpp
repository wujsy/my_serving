#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/FRCNN/util/frcnn_vis.hpp"
#include "caffe/FRCNN/frcnn_api.hpp"

DEFINE_string(gpu, "", 
    "Optional; run in GPU mode on the given device ID, Empty is CPU");
DEFINE_string(model, "", 
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "", 
    "Trained Model By Faster RCNN End-to-End Pipeline.");
DEFINE_string(default_c, "", 
    "Default config file path.");
DEFINE_string(in_image, "",
    "Optional;Test images root directory.");
DEFINE_int32(num_image, 1,
    "Optional;Test images number.");
DEFINE_string(out_file, "", 
    "Optional;Output images file.");


inline std::string INT(float x) { char A[100]; sprintf(A,"%.1f",x); return std::string(A);};
inline std::string FloatToString(float x) { char A[100]; sprintf(A,"%.4f",x); return std::string(A);};

int main(int argc, char** argv){
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          7       use 7-th gpu device, default is cpu model\n"
      "  --model        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --default_c    file    Default Config File\n"
      "  --in_image     file    input image file name\n"
      "  --num_image    int     input image num\n"
      "  --out_file     file    output amswer file");

  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  CHECK( FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1 ) << "Can only support one gpu or none";
  int gpu_id = -1;
  if( FLAGS_gpu.size() > 0 )
    gpu_id = boost::lexical_cast<int>(FLAGS_gpu);  

  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
    LOG(FATAL) << "CPU ONLY MODEL, BUT PROVIDE GPU ID";
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

#ifdef _THINKPAD_
  const int num_image = 3;
  //std::string proto_file = "/home/ypzhang/workspace/autohome/work/other/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end_2class_prune/test.prototxt";
  //std::string model_file = "/home/ypzhang/workspace/autohome/work/other/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_end2end_2class_prune/vgg16_faster_rcnn_2class_prune_iter_70000.caffemodel";
  std::string proto_file             = "/home/ypzhang/Desktop/debug_zf_prune/auto_test.prototxt";
  std::string model_file             = "/home/ypzhang/Desktop/debug_zf_prune/zf_faster_rcnn_2class_prune_iter_70000.caffemodel";
  std::string default_config_file    = "/home/ypzhang/workspace/autohome/work/online_serving/Tools/auto_caffe/examples/FRCNN/config/voc_config_2class.json";

  const std::string in_file = "/home/ypzhang/workspace/autohome/work/online_serving/Data/";
  //const std::string image1 = "/home/ypzhang/workspace/autohome/work/online_serving/Data/more_cars.jpg";
  //const std::string image2 = "/home/ypzhang/workspace/autohome/work/online_serving/Data/images.jpg";
  //const std::string image3 = "/home/ypzhang/workspace/autohome/work/online_serving/Data/000456.jpg";
  const std::string out_file = "/home/ypzhang/Desktop/try";
#else
  const int num_image = FLAGS_num_image;
  CHECK_GE(num_image, 1) << "Need > 1 input image!";

  std::string proto_file             = FLAGS_model.c_str();
  std::string model_file             = FLAGS_weights.c_str();
  std::string default_config_file    = FLAGS_default_c.c_str();

  const std::string in_file = FLAGS_in_image.c_str();
  const std::string out_file = FLAGS_out_file.c_str();
#endif

  LOG(INFO) << "proto_file: " << proto_file;
  LOG(INFO) << "model_file: " << model_file;
  LOG(INFO) << "in_file: " << in_file;
  LOG(INFO) << "out_file: " << out_file;
  FRCNN_API::Detector detector(proto_file, model_file, default_config_file);

  /// new
  /*for(int i = 0; i < 100; i++){
      cv::Mat cv_image1 = cv::imread(image_list);
      std::vector<caffe::Frcnn::BBox<float> > results1;
      detector.predict(cv_image1, results1);
  }*/

  std::vector<cv::Size > orig_size;
  std::vector<cv::Mat > input_images;
  for(int i = 0; i < num_image; i++){
      std::stringstream filenm;
      filenm << in_file << i << ".jpg";

      cv::Mat img = cv::imread(filenm.str());
      orig_size.push_back(img.size());
      if(i > 0)
          cv::resize(img, img, input_images.at(0).size());
      input_images.push_back(img);
  }

  std::vector<std::vector<caffe::Frcnn::BBox<float> > > results;
  detector.predict(input_images, results);

  CHECK_EQ(results.size(), num_image);

  for(int img = 0; img < num_image; img++){
      std::vector<caffe::Frcnn::BBox<float> > per_result = results.at(img);
      cv::Mat image = input_images.at(img);
      for(int ir = 0; ir < per_result.size(); ir++){
          if(per_result[ir].confidence > 0.9){
              cv::Rect rect(per_result[ir][0], per_result[ir][1], per_result[ir][2], per_result[ir][3]);
              ///cv::rectangle(cv_image, rect, cv::Scalar(255, 0, 0), 2);
              cv::rectangle(image, cv::Point(per_result[ir][0],per_result[ir][1])
                           , cv::Point(per_result[ir][2],per_result[ir][3]), cv::Scalar(255, 0, 0));

              std::ostringstream text;
              text << per_result[ir].confidence;
              cv::putText(image
                          , text.str()
                          , cv::Point(rect.x, rect.y)
                          , CV_FONT_HERSHEY_COMPLEX
                          , 0.8
                          , cv::Scalar(0, 255, 0));
          }
      }
      cv::resize(image, image, orig_size.at(img));
      std::stringstream filenm;
      filenm << out_file << img << ".jpg";
      cv::imwrite(filenm.str(), image);
  }
  return 0;
}
