#!/bin/bash

# Note: The instructions here correspond to an Ubuntu Linux environment; 
# although some commands may differ for other operating systems and distributions, 
# the general ideas are identical.

# install dependencies
# Protobuf Dependencies
sudo apt-get install autoconf automake libtool curl make g++ unzip
# TensorFlow Dependencies
sudo apt-get install python-numpy swig python-dev python-wheel
# install tensorflow r1.2
git clone --recurse-submodules -b r1.2 https://github.com/tensorflow/tensorflow.git

# add build rules to BUILD file
cd tensorflow/
build_file="tensorflow/BUILD"
build_rule="
# Added build rule
cc_binary(
    name = "libtensorflow_all.so",
    linkshared = 1,
    linkopts = ["-Wl,--version-script=tensorflow/tf_version_script.lds"], #if use Mac remove this line
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
    ],
)
"
if [-w $build_file]
then 
  echo "$build_rule" >> $build_file
else
  echo "$build_file has no written permissions"
fi

# configure tensorflow and compile tensorflow shared library
# note that choose 'Y' when ask whether need CUDA support
./configure
export TF_NEED_CUDA=1
# This specifies a new build rule, producing libtensorflow_all.so, that includes 
# all the required dependencies for integration with a C++ project
bazel build -c opt --config=cuda tensorflow:libtensorflow_all.so

# Copy the source to specific dir and remove unneeded items:
sudo cp bazel-bin/tensorflow/libtensorflow_all.so /usr/local/lib
sudo mkdir -p /usr/local/include/google/tensorflow
sudo cp -r tensorflow /usr/local/include/google/tensorflow/
sudo find /usr/local/include/google/tensorflow/tensorflow -type f  ! -name "*.h" -delete

# Copy all generated files from bazel-genfiles
sudo cp bazel-genfiles/tensorflow/core/framework/*.h  /usr/local/include/google/tensorflow/tensorflow/core/framework
sudo cp bazel-genfiles/tensorflow/core/kernels/*.h  /usr/local/include/google/tensorflow/tensorflow/core/kernels
sudo cp bazel-genfiles/tensorflow/core/lib/core/*.h  /usr/local/include/google/tensorflow/tensorflow/core/lib/core
sudo cp bazel-genfiles/tensorflow/core/protobuf/*.h  /usr/local/include/google/tensorflow/tensorflow/core/protobuf
sudo cp bazel-genfiles/tensorflow/core/util/*.h  /usr/local/include/google/tensorflow/tensorflow/core/util
sudo cp bazel-genfiles/tensorflow/cc/ops/*.h  /usr/local/include/google/tensorflow/tensorflow/cc/ops

# Copy the third party directory
sudo cp -r third_party /usr/local/include/google/tensorflow/
sudo rm -r /usr/local/include/google/tensorflow/third_party/py
# Note: newer versions of TensorFlow do not have the following directory
#sudo rm -r /usr/local/include/google/tensorflow/third_party/avro

# --------------------------------------------------------------------------------------
# Copy Eigen and Protobuf source from external directory
sudo cp -r bazel-tensorflow/external/eigen_archive /usr/local/include/google/tensorflow/third_party/
sudo cp -r bazel-tensorflow/external/eigen_archive/protobuf/src /usr/local/include/google/tensorflow/third_party/

# Copy genfiles and org_tensorflow files 
sudo cp -r bazel-out/local_linux-opt/genfiles /usr/local/include/google/tensorflow/
sudo cp -r bazel-tensorflow /usr/local/include/google/tensorflow/
















