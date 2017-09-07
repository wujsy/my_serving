
#include "cv_process.hpp"
#include <fstream>

bool cvprocess::flip(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    cv::flip(src, dst, 0);

    return writeImage(dst, odata);
}

bool cvprocess::medianBlur(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    cv::medianBlur(src, dst, 3);

    return writeImage(dst, odata);
}

bool cvprocess::applyColorMap(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    //cv::applyColorMap(src, dst, 0);

    return writeImage(dst, odata);
}

bool cvprocess::cvtColor(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    cv::cvtColor(src, dst, CV_BGR2GRAY);

    return writeImage(dst, odata);
}


bool cvprocess::readImage(std::vector<unsigned char>& idata, cv::Mat& dst) {
    if (idata.size() == 0)
        return false;

    dst = cv::imdecode(idata, CV_LOAD_IMAGE_UNCHANGED);
    //dst = cv::imdecode(idata, IMREAD_UNCHANGED);

    return dst.data != nullptr;
}

bool cvprocess::writeImage(cv::Mat& src, std::vector<unsigned char>& odata) {
    if (src.data == nullptr)
        return false;

    vector<int> param;
    param.push_back(IMWRITE_PNG_COMPRESSION);
    param.push_back(3); //default(3) 0-9.
    return cv::imencode(".png", src, odata, param);
}

bool cvprocess::resizeImage(cv::Mat& input, float bmeans, float gmeans, float rmeans, int ewidth, int eheight, cv::Mat& img){
    if (input.data == nullptr)
        return false;

//cv::Mat img;
        const int height = input.rows;
        const int width = input.cols;
        input.convertTo(img, CV_32FC3);
        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                int offset = (r * img.cols + c) * 3;
                reinterpret_cast<float *>(img.data)[offset + 0] -= bmeans; // B
                reinterpret_cast<float *>(img.data)[offset + 1] -= gmeans;
                reinterpret_cast<float *>(img.data)[offset + 2] -= rmeans;
            }
        }

float scale_factor_w = (float)ewidth / width;
float scale_factor_h = (float)eheight / height;
        cv::resize(img, img, cv::Size(), scale_factor_w, scale_factor_h);

}



/*
bool cvprocess::process_tf_detect_result(std::vector<deep_server::Tf_detect_result> results, cv::Mat& cv_image) {
    for (int ir = 0; ir < results.size(); ir++) {
        //if (results[ir].confidence > 0.9
            ) 
        {
            cv::Rect rect(results[ir].x(), results[ir].y(), results[ir].w(), results[ir].h());
           // cv::rectangle(cv_image, rect, cv::Scalar(255, 0, 0), 2);

           cv::rectangle(cv_image, cv::Point(results[ir].x(), results[ir].y())
                       , cv::Point(results[ir].w(), results[ir].h()), cv::Scalar(255, 0, 0), 2);
            char text[64];
            //sprintf(text, "%lf", results[ir].confidence);
            cv::putText(cv_image
                , text
                , cv::Point(rect.x, rect.y)
                , CV_FONT_HERSHEY_COMPLEX
                , 0.8
                , cv::Scalar(0, 0, 255));
        }
    }

    return true;
}
*/
