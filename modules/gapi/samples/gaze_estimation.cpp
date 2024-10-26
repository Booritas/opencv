#include <algorithm>
#include <iostream>
#include <cctype>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/highgui.hpp> // CommandLineParser
#include <opencv2/gapi/infer/parsers.hpp>

const std::string about =
    "This is an OpenCV-based version of Gaze Estimation example";
const std::string keys =
    "{ h help |                                    | Print this help message }"
    "{ input  |                                    | Path to the input video file }"
    "{ facem  | face-detection-retail-0005.xml     | Path to OpenVINO face detection model (.xml) }"
    "{ faced  | CPU                                | Target device for the face detection (e.g. CPU, GPU, ...) }"
    "{ landm  | facial-landmarks-35-adas-0002.xml  | Path to OpenVINO landmarks detector model (.xml) }"
    "{ landd  | CPU                                | Target device for the landmarks detector (e.g. CPU, GPU, ...) }"
    "{ headm  | head-pose-estimation-adas-0001.xml | Path to OpenVINO head pose estimation model (.xml) }"
    "{ headd  | CPU                                | Target device for the head pose estimation inference (e.g. CPU, GPU, ...) }"
    "{ gazem  | gaze-estimation-adas-0002.xml      | Path to OpenVINO gaze vector estimaiton model (.xml) }"
    "{ gazed  | CPU                                | Target device for the gaze vector estimation inference (e.g. CPU, GPU, ...) }"
    ;

namespace {
std::string weights_path(const std::string &model_path) {
    const auto EXT_LEN = 4u;
    const auto sz = model_path.size();
    CV_Assert(sz > EXT_LEN);

    auto ext   = model_path.substr(sz - EXT_LEN);
    auto lower = [](unsigned char c) {
        return static_cast<unsigned char>(std::tolower(c));
    };
    std::transform(ext.begin(), ext.end(), ext.begin(), lower);
    CV_Assert(ext == ".xml");
    return model_path.substr(0u, sz - EXT_LEN) + ".bin";
}
} // anonymous namespace

namespace custom {
namespace {
using GMat3  = std::tuple<ncvslideio::GMat,ncvslideio::GMat,ncvslideio::GMat>;
using GMats  = ncvslideio::GArray<ncvslideio::GMat>;
using GRects = ncvslideio::GArray<ncvslideio::Rect>;
using GSize  = ncvslideio::GOpaque<ncvslideio::Size>;
G_API_NET(Faces,     <ncvslideio::GMat(ncvslideio::GMat)>, "face-detector"   );
G_API_NET(Landmarks, <ncvslideio::GMat(ncvslideio::GMat)>, "facial-landmarks");
G_API_NET(HeadPose,  <   GMat3(ncvslideio::GMat)>, "head-pose");
G_API_NET(Gaze,      <ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GMat,ncvslideio::GMat)>, "gaze-vector");

G_API_OP(Size, <GSize(ncvslideio::GMat)>, "custom.gapi.size") {
    static ncvslideio::GOpaqueDesc outMeta(const ncvslideio::GMatDesc &) {
        return ncvslideio::empty_gopaque_desc();
    }
};

// Left/Right eye per every face
G_API_OP(ParseEyes,
         <std::tuple<GRects, GRects>(GMats, GRects, GSize)>,
         "custom.gaze_estimation.parseEyes") {
    static std::tuple<ncvslideio::GArrayDesc, ncvslideio::GArrayDesc>
        outMeta(  const ncvslideio::GArrayDesc &
                , const ncvslideio::GArrayDesc &
                , const ncvslideio::GOpaqueDesc &) {
        return std::make_tuple(ncvslideio::empty_array_desc(), ncvslideio::empty_array_desc());
    }
};

// Combine three scalars into a 1x3 vector (per every face)
G_API_OP(ProcessPoses,
         <GMats(GMats, GMats, GMats)>,
         "custom.gaze_estimation.processPoses") {
    static ncvslideio::GArrayDesc outMeta(  const ncvslideio::GArrayDesc &
                                  , const ncvslideio::GArrayDesc &
                                  , const ncvslideio::GArrayDesc &) {
        return ncvslideio::empty_array_desc();
    }
};

void gazeVectorToGazeAngles(const ncvslideio::Point3f& gazeVector,
                                  ncvslideio::Point2f& gazeAngles) {
    auto r = ncvslideio::norm(gazeVector);

    double v0 = static_cast<double>(gazeVector.x);
    double v1 = static_cast<double>(gazeVector.y);
    double v2 = static_cast<double>(gazeVector.z);

    gazeAngles.x = static_cast<float>(180.0 / M_PI * (M_PI_2 + std::atan2(v2, v0)));
    gazeAngles.y = static_cast<float>(180.0 / M_PI * (M_PI_2 - std::acos(v1 / r)));
}

GAPI_OCV_KERNEL(OCVSize, Size) {
    static void run(const ncvslideio::Mat &in, ncvslideio::Size &out) {
        out = in.size();
    }
};

ncvslideio::Rect eyeBox(const ncvslideio::Rect &face_rc,
                float p1_x, float p1_y, float p2_x, float p2_y,
                float scale = 1.8f) {
    const auto &up = face_rc.size();
    const ncvslideio::Point p1 = {
        static_cast<int>(p1_x*up.width),
        static_cast<int>(p1_y*up.height)
    };
    const ncvslideio::Point p2 = {
        static_cast<int>(p2_x*up.width),
        static_cast<int>(p2_y*up.height)
    };
    ncvslideio::Rect result;

    const auto size     = static_cast<float>(ncvslideio::norm(p1 - p2));
    const auto midpoint = (p1 + p2) / 2;

    result.width = static_cast<int>(scale * size);
    result.height = result.width;
    result.x = face_rc.x + midpoint.x - (result.width / 2);
    result.y = face_rc.y + midpoint.y - (result.height / 2);
    // Shift result to the original frame's absolute coordinates
    return result;
}

GAPI_OCV_KERNEL(OCVParseEyes, ParseEyes) {
    static void run(const std::vector<ncvslideio::Mat> &in_landmarks_per_face,
                    const std::vector<ncvslideio::Rect> &in_face_rcs,
                    const ncvslideio::Size &frame_size,
                    std::vector<ncvslideio::Rect> &out_left_eyes,
                    std::vector<ncvslideio::Rect> &out_right_eyes) {
        const size_t numFaces = in_landmarks_per_face.size();
        const ncvslideio::Rect surface(ncvslideio::Point(0,0), frame_size);
        GAPI_Assert(numFaces == in_face_rcs.size());
        out_left_eyes.clear();
        out_right_eyes.clear();
        out_left_eyes.reserve(numFaces);
        out_right_eyes.reserve(numFaces);

        for (std::size_t i = 0u; i < numFaces; i++) {
            const auto &lm = in_landmarks_per_face[i];
            const auto &rc = in_face_rcs[i];
            // Left eye is defined by points 0/1 (x2),
            // Right eye is defined by points 2/3 (x2)
            const float *data = lm.ptr<float>();
            out_left_eyes .push_back(surface & eyeBox(rc, data[0], data[1], data[2], data[3]));
            out_right_eyes.push_back(surface & eyeBox(rc, data[4], data[5], data[6], data[7]));
        }
    }
};

GAPI_OCV_KERNEL(OCVProcessPoses, ProcessPoses) {
    static void run(const std::vector<ncvslideio::Mat> &in_ys,
                    const std::vector<ncvslideio::Mat> &in_ps,
                    const std::vector<ncvslideio::Mat> &in_rs,
                    std::vector<ncvslideio::Mat> &out_poses) {
        const std::size_t sz = in_ys.size();
        GAPI_Assert(sz == in_ps.size() && sz == in_rs.size());
        out_poses.clear();
        for (std::size_t idx = 0u; idx < sz; idx++) {
            ncvslideio::Mat pose(1, 3, CV_32FC1);
            float *ptr = pose.ptr<float>();
            ptr[0] = in_ys[idx].ptr<float>()[0];
            ptr[1] = in_ps[idx].ptr<float>()[0];
            ptr[2] = in_rs[idx].ptr<float>()[0];
            out_poses.push_back(std::move(pose));
        }
    }
};
} // anonymous namespace
} // namespace custom

namespace vis {
namespace {
ncvslideio::Point2f midp(const ncvslideio::Rect &rc) {
    return (rc.tl() + rc.br()) / 2;
};
void bbox(ncvslideio::Mat &m, const ncvslideio::Rect &rc) {
    ncvslideio::rectangle(m, rc, ncvslideio::Scalar{0,255,0}, 2, ncvslideio::LINE_8, 0);
};
void pose(ncvslideio::Mat &m, const ncvslideio::Mat &p, const ncvslideio::Rect &face_rc) {
    const auto *posePtr = p.ptr<float>();
    const auto yaw   = static_cast<double>(posePtr[0]);
    const auto pitch = static_cast<double>(posePtr[1]);
    const auto roll  = static_cast<double>(posePtr[2]);

    const auto sinY = std::sin(yaw   * M_PI / 180.0);
    const auto sinP = std::sin(pitch * M_PI / 180.0);
    const auto sinR = std::sin(roll  * M_PI / 180.0);

    const auto cosY = std::cos(yaw   * M_PI / 180.0);
    const auto cosP = std::cos(pitch * M_PI / 180.0);
    const auto cosR = std::cos(roll  * M_PI / 180.0);

    const auto axisLength = 0.4 * face_rc.width;
    const auto xCenter = face_rc.x + face_rc.width  / 2;
    const auto yCenter = face_rc.y + face_rc.height / 2;

    const auto center = ncvslideio::Point{xCenter, yCenter};
    const auto axisln = ncvslideio::Point2d{axisLength, axisLength};
    const auto ctr    = ncvslideio::Matx<double,2,2>(cosR*cosY, sinY*sinP*sinR, 0.f,  cosP*sinR);
    const auto ctt    = ncvslideio::Matx<double,2,2>(cosR*sinY*sinP, cosY*sinR, 0.f, -cosP*cosR);
    const auto ctf    = ncvslideio::Matx<double,2,2>(sinY*cosP, 0.f, 0.f, sinP);

    // center to right
    ncvslideio::line(m, center, center + static_cast<ncvslideio::Point>(ctr*axisln), ncvslideio::Scalar(0, 0, 255), 2);
    // center to top
    ncvslideio::line(m, center, center + static_cast<ncvslideio::Point>(ctt*axisln), ncvslideio::Scalar(0, 255, 0), 2);
    // center to forward
    ncvslideio::line(m, center, center + static_cast<ncvslideio::Point>(ctf*axisln), ncvslideio::Scalar(255, 0, 255), 2);
}
void vvec(ncvslideio::Mat &m, const ncvslideio::Mat &v, const ncvslideio::Rect &face_rc,
          const ncvslideio::Rect &left_rc, const ncvslideio::Rect &right_rc) {
    const auto scale =  0.002 * face_rc.width;

    ncvslideio::Point3f gazeVector;
    const auto *gazePtr = v.ptr<float>();
    gazeVector.x = gazePtr[0];
    gazeVector.y = gazePtr[1];
    gazeVector.z = gazePtr[2];
    gazeVector = gazeVector / ncvslideio::norm(gazeVector);

    const double arrowLength = 0.4 * face_rc.width;
    const auto left_mid = midp(left_rc);
    const auto right_mid = midp(right_rc);

    ncvslideio::Point2f gazeArrow;
    gazeArrow.x =  gazeVector.x;
    gazeArrow.y = -gazeVector.y;
    gazeArrow  *= arrowLength;

    ncvslideio::arrowedLine(m, left_mid,  left_mid  + gazeArrow, ncvslideio::Scalar(255, 0, 0), 2);
    ncvslideio::arrowedLine(m, right_mid, right_mid + gazeArrow, ncvslideio::Scalar(255, 0, 0), 2);

    ncvslideio::Point2f gazeAngles;
    custom::gazeVectorToGazeAngles(gazeVector, gazeAngles);

    ncvslideio::putText(m,
                ncvslideio::format("gaze angles: (h=%0.0f, v=%0.0f)",
                           static_cast<double>(std::round(gazeAngles.x)),
                           static_cast<double>(std::round(gazeAngles.y))),
                ncvslideio::Point(static_cast<int>(face_rc.tl().x),
                          static_cast<int>(face_rc.br().y + 12. * face_rc.width / 100.)),
                ncvslideio::FONT_HERSHEY_PLAIN, scale * 2, ncvslideio::Scalar::all(255), 1);
};
} // anonymous namespace
} // namespace vis

int main(int argc, char *argv[])
{
    ncvslideio::CommandLineParser cmd(argc, argv, keys);
    cmd.about(about);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }
    ncvslideio::GMat in;
    ncvslideio::GMat faces = ncvslideio::gapi::infer<custom::Faces>(in);
    ncvslideio::GOpaque<ncvslideio::Size> sz = ncvslideio::gapi::streaming::size(in);
    ncvslideio::GArray<ncvslideio::Rect> faces_rc = ncvslideio::gapi::parseSSD(faces, sz, 0.5f, true, true);
    ncvslideio::GArray<ncvslideio::GMat> angles_y, angles_p, angles_r;
    std::tie(angles_y, angles_p, angles_r) = ncvslideio::gapi::infer<custom::HeadPose>(faces_rc, in);
    ncvslideio::GArray<ncvslideio::GMat> heads_pos = custom::ProcessPoses::on(angles_y, angles_p, angles_r);
    ncvslideio::GArray<ncvslideio::GMat> landmarks = ncvslideio::gapi::infer<custom::Landmarks>(faces_rc, in);
    ncvslideio::GArray<ncvslideio::Rect> left_eyes, right_eyes;
    std::tie(left_eyes, right_eyes) = custom::ParseEyes::on(landmarks, faces_rc, sz);
    ncvslideio::GArray<ncvslideio::GMat> gaze_vectors = ncvslideio::gapi::infer2<custom::Gaze>( in
                                                                      , left_eyes
                                                                      , right_eyes
                                                                      , heads_pos);
    ncvslideio::GComputation graph(ncvslideio::GIn(in),
                           ncvslideio::GOut( ncvslideio::gapi::copy(in)
                                   , faces_rc
                                   , left_eyes
                                   , right_eyes
                                   , heads_pos
                                   , gaze_vectors));

    const auto input_file_name = cmd.get<std::string>("input");
    const auto face_model_path = cmd.get<std::string>("facem");
    const auto head_model_path = cmd.get<std::string>("headm");
    const auto lmrk_model_path = cmd.get<std::string>("landm");
    const auto gaze_model_path = cmd.get<std::string>("gazem");

    auto face_net = ncvslideio::gapi::ie::Params<custom::Faces> {
        face_model_path,                // path to topology IR
        weights_path(face_model_path),  // path to weights
        cmd.get<std::string>("faced"),  /// device specifier
    };
    auto head_net = ncvslideio::gapi::ie::Params<custom::HeadPose> {
        head_model_path,                // path to topology IR
        weights_path(head_model_path),  // path to weights
        cmd.get<std::string>("headd"),  // device specifier
    }.cfgOutputLayers({"angle_y_fc", "angle_p_fc", "angle_r_fc"});
    auto landmarks_net = ncvslideio::gapi::ie::Params<custom::Landmarks> {
        lmrk_model_path,                // path to topology IR
        weights_path(lmrk_model_path),  // path to weights
        cmd.get<std::string>("landd"),  // device specifier
    };
    auto gaze_net = ncvslideio::gapi::ie::Params<custom::Gaze> {
        gaze_model_path,                // path to topology IR
        weights_path(gaze_model_path),  // path to weights
        cmd.get<std::string>("gazed"),  // device specifier
    }.cfgInputLayers({"left_eye_image", "right_eye_image", "head_pose_angles"});

    auto kernels = ncvslideio::gapi::kernels< custom::OCVSize
                                    , custom::OCVParseEyes
                                    , custom::OCVProcessPoses>();
    auto networks = ncvslideio::gapi::networks(face_net, head_net, landmarks_net, gaze_net);
    auto pipeline = graph.compileStreaming(ncvslideio::compile_args(networks, kernels));

    ncvslideio::TickMeter tm;
    ncvslideio::Mat image;
    std::vector<ncvslideio::Rect> out_faces, out_right_eyes, out_left_eyes;
    std::vector<ncvslideio::Mat> out_poses;
    std::vector<ncvslideio::Mat> out_gazes;
    std::size_t frames = 0u;
    std::cout << "Reading " << input_file_name << std::endl;

    pipeline.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(input_file_name));
    pipeline.start();
    tm.start();
    while (pipeline.pull(ncvslideio::gout( image
                                 , out_faces
                                 , out_left_eyes
                                 , out_right_eyes
                                 , out_poses
                                 , out_gazes))) {
        frames++;
        // Visualize results on the frame
        for (auto &&rc : out_faces) vis::bbox(image, rc);
        for (auto &&rc : out_left_eyes) vis::bbox(image, rc);
        for (auto &&rc : out_right_eyes) vis::bbox(image, rc);
        for (std::size_t i = 0u; i < out_faces.size(); i++) {
            vis::pose(image, out_poses[i], out_faces[i]);
            vis::vvec(image, out_gazes[i], out_faces[i], out_left_eyes[i], out_right_eyes[i]);
        }
        tm.stop();
        const auto fps_str = std::to_string(frames / tm.getTimeSec()) + " FPS";
        ncvslideio::putText(image, fps_str, {0,32}, ncvslideio::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        ncvslideio::imshow("Out", image);
        ncvslideio::waitKey(1);
        tm.start();
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames"
              << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
