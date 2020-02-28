#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> 
#include "opencv2/core/utility.hpp"

template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    return cv::samples::findFile(std::string("../opencv_extra/testdata/dnn/") + filename, required);
}

int main(int argc, const char* argv[])
{
    cv::Mat input = cv::imread(_tf("pose.png"), 0);
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(input, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::namedWindow( "Display window 2", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window 2", output );                   // Show our image inside it.

    cv::waitKey(0);                             
    return 0;
}
