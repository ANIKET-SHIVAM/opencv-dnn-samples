#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgcodecs.hpp> 
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/dnn.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;


dnn::Backend backend = dnn::Backend::DNN_BACKEND_DEFAULT;
dnn::Target target = dnn::Target::DNN_TARGET_CPU;
template<typename TString>
static std::string _tf(TString filename, bool required = true)
{
    return samples::findFile(std::string("../opencv_extra/testdata/dnn/") + filename, required);
}

static std::string getType(const std::string& header)
{
    std::string field = "'descr':";
    int idx = header.find(field);
    CV_Assert(idx != -1);

    int from = header.find('\'', idx + field.size()) + 1;
    int to = header.find('\'', from);
    return header.substr(from, to - from);
}

static std::string getFortranOrder(const std::string& header)
{
    std::string field = "'fortran_order':";
    int idx = header.find(field);
    CV_Assert(idx != -1);

    int from = header.find_last_of(' ', idx + field.size()) + 1;
    int to = header.find(',', from);
    return header.substr(from, to - from);
}

static std::vector<int> getShape(const std::string& header)
{
    std::string field = "'shape':";
    int idx = header.find(field);
    CV_Assert(idx != -1);

    int from = header.find('(', idx + field.size()) + 1;
    int to = header.find(')', from);
    
    std::string shapeStr = header.substr(from, to - from);
    if (shapeStr.empty())
        return std::vector<int>(1, 1);
    
    // Remove all commas.
    shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), ','),
                   shapeStr.end());
  
    std::istringstream ss(shapeStr);
    int value;

    std::vector<int> shape;
    while (ss >> value)
    {
        shape.push_back(value);
    }
    return shape;
}

Mat blobFromNPY(const std::string& path)
{
    std::ifstream ifs(path.c_str(), std::ios::binary);
    CV_Assert(ifs.is_open());

    std::string magic(6, '*');
    ifs.read(&magic[0], magic.size());
    CV_Assert(magic == "\x93NUMPY");

    ifs.ignore(1);  // Skip major version byte.
    ifs.ignore(1);  // Skip minor version byte.

    unsigned short headerSize;
    ifs.read((char*)&headerSize, sizeof(headerSize));

    std::string header(headerSize, '*');
    ifs.read(&header[0], header.size());

    // Extract data type.
    CV_Assert(getType(header) == "<f4");
    CV_Assert(getFortranOrder(header) == "False");
    std::vector<int> shape = getShape(header);

    Mat blob(shape, CV_32F);
    ifs.read((char*)blob.data, blob.total() * blob.elemSize());
    CV_Assert((size_t)ifs.gcount() == blob.total() * blob.elemSize());

    return blob;
}



std::vector<Point2f> testKeyPointsModel(const std::string& weights, const std::string& cfg,
                        const Mat& frame, const Mat& exp, float norm,
                        const Size& size = {-1, -1}, Scalar mean = Scalar(),
                        double scale = 1.0, bool swapRB = false, bool crop = false)
{
    

    std::vector<Point2f> points;

    dnn::KeypointsModel model(weights, cfg);
    model.setInputSize(size).setInputMean(mean).setInputScale(scale)
         .setInputSwapRB(swapRB).setInputCrop(crop);

    model.setPreferableBackend(backend);
    model.setPreferableTarget(target);

    points = model.estimate(frame, 0.5);

    return points;
}


std::vector<Point2f> dnn_keypoint_pose()
{
    Mat inp = cv::imread(_tf("pose_test2.jpeg"));
    std::string weights = _tf("onnx/models/lightweight_pose_estimation.onnx", false);
    Mat exp = blobFromNPY(_tf("keypoints_exp.npy"));


    Size size{507, 626};
    float norm = 1e-4;
    double scale = 1.0/255;
    Scalar mean = Scalar(128, 128, 128);
    bool swapRB = false;

    // Ref. Range: [58.6875, 508.625]
    if (target == dnn::Target::DNN_TARGET_CUDA_FP16)
        norm = 20; // l1 = 1.5, lInf = 20

    std::vector<Point2f> keypointMat = testKeyPointsModel(weights, "", inp, exp, norm, size, mean, scale, swapRB);
    return keypointMat;
}

std::vector<Point2f> dnn_keypoint_facial()
{
    Mat inp = cv::imread(_tf("michelle_detected.png"), 0); 
    std::string weights = _tf("onnx/models/facial_keypoints.onnx", false);
    Mat exp = blobFromNPY(_tf("facial_keypoints_exp.npy"));

    Size size{227, 227};
    float norm = (target == dnn::Target::DNN_TARGET_OPENCL_FP16) ? 5e-3 : 1e-4;
    double scale = 1.0/255;
    Scalar mean = Scalar();
    bool swapRB = false;

    // Ref. Range: [-1.1784188, 1.7758257]
    if (target == dnn::Target::DNN_TARGET_CUDA_FP16)
        norm = 0.004; // l1 = 0.0006, lInf = 0.004

    std::vector<Point2f> keypointMat = testKeyPointsModel(weights, "", inp, exp, norm, size, mean, scale, swapRB);
    return keypointMat;
}

int main () {
  std::vector<Point2f> keypointMatPose, keypointMatFace;
  for (int i = 0; i < 1; i++) {
    keypointMatPose = dnn_keypoint_pose();
    keypointMatFace = dnn_keypoint_facial();
  }
/*    Mat image;
    image = imread(_tf("pose_test2.jpeg"), IMREAD_COLOR);   // Read the file
    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.
    Mat image2;
    image2 = imread(_tf("michelle_detected.png",0), IMREAD_COLOR);   // Read the file
    if(! image2.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    namedWindow( "Display window 2", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window 2", image2 );                   // Show our image inside it.
*/
  std::cout << "Pose keypoints = " << std::endl << " "  << keypointMatPose << std::endl << std::endl;
  std::cout << "Facial keypoints = " << std::endl << " "  << keypointMatFace << std::endl << std::endl;
  Mat outPose, outFace;
  std::vector<cv::KeyPoint> keypointsPose;

  for( size_t i = 0; i < keypointMatPose.size(); i++ ) 
    keypointsPose.push_back(KeyPoint(keypointMatPose[i], 1.f));
  
  Mat faceimage = imread(_tf("michelle_detected.png",0), 0);
  std::cout << "Image size: " <<  faceimage.cols << "," << faceimage.rows << std::endl;

  for( size_t i = 0; i < keypointMatFace.size(); i++ ) {
    keypointMatFace[i].x = (keypointMatFace[i].x * 50 + 100) * ((float)faceimage.cols/227);
    keypointMatFace[i].y = (keypointMatFace[i].y * 50 + 100) * ((float)faceimage.rows/227);
  }
  std::cout << "Facial keypoints = " << std::endl << " "  << keypointMatFace << std::endl << std::endl;

  std::vector<cv::KeyPoint> keypointsFace;
  for( size_t i = 0; i < keypointMatFace.size(); i++ ) 
    keypointsFace.push_back(KeyPoint(keypointMatFace[i], 1.f));

  drawKeypoints(imread(_tf("pose_test2.jpeg"), 0),keypointsPose,outPose); 
  namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window", outPose );                   // Show our image inside it.
  drawKeypoints(faceimage,keypointsFace,outFace);
  namedWindow( "Display window 2", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window 2", outFace );                   // Show our image inside it.
  
  waitKey(0);                             
  return 0;
}
