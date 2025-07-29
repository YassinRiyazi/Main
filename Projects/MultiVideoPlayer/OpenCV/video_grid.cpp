// main.cpp
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main()
{
    // --- 1. Gather last 15 .mp4 paths ---
    std::vector<fs::path> all;
    for (auto &p : fs::recursive_directory_iterator("/media/d2u25/Dont/S4S-ROF/frame_Extracted")) {
        if (p.path().extension() == ".mp4")
            all.push_back(p.path());
    }
    // sort by filename[1] as per original (odd, but preserving)
    std::sort(all.begin(), all.end(),
              [](auto &a, auto &b){ return a.filename().string()[1] < b.filename().string()[1]; });
    // take last 15
    std::vector<fs::path> vids(all.end() - std::min<size_t>(15, all.size()), all.end());

    // --- 2. Open VideoCaptures ---
    std::vector<cv::VideoCapture> caps;
    std::vector<int> totalFrames;
    std::vector<std::string> labels;
    for (auto &p : vids) {
        cv::VideoCapture cap(p.string(), cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            std::cerr << "Error: cannot open " << p << "\n";
            return -1;
        }
        caps.push_back(std::move(cap));
        int tf = int(caps.back().get(cv::CAP_PROP_FRAME_COUNT));
        totalFrames.push_back(tf);
        labels.push_back(p.filename().string());
    }

    // --- 3. Grid params ---
    const int ROW = 3, COL = 5;
    int height = (1920-100)/COL;
    int width = (1200-100)/ROW;

    const cv::Size OUTSZ(width,height);
    const cv::Scalar BLACK(0,0,0);
    cv::Mat blank(OUTSZ, CV_8UC3, BLACK);

    // Pre-allocate frame mats
    std::vector<cv::Mat> frames(ROW*COL, blank);

    // --- 4. GUI state ---
    bool paused = false, showNames = false, showProg = false;
    const std::string win = std::to_string(ROW) + "x" + std::to_string(COL) + " Video Grid";
    cv::namedWindow(win, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    const auto font = cv::FONT_HERSHEY_SIMPLEX;

    while (true)
    {
        if (!paused) {
            for (int i = 0; i < (int)caps.size(); ++i) {
                cv::Mat f;
                if (!caps[i].read(f) || f.empty()) {
                    frames[i] = blank;
                } else {
                    // resize if needed
                    if (f.size() != OUTSZ) cv::resize(f, f, OUTSZ, 0, 0, cv::INTER_LINEAR);

                    if (showNames) {
                        cv::putText(f, labels[i], {10,30}, font, 0.6, cv::Scalar(0,0,255), 1, cv::LINE_AA);
                    }
                    if (showProg) {
                        int pos = int(caps[i].get(cv::CAP_PROP_POS_FRAMES));
                        int total = totalFrames[i];
                        int pct = total>0 ? 100 - (pos*100/total) : 0;
                        std::string txt = std::to_string(pct) + "% left";
                        cv::putText(f, txt, {10,55}, font, 0.6, cv::Scalar(0,255,0), 1, cv::LINE_AA);
                    }
                    frames[i] = f;
                }
            }
        }

        // assemble grid
        std::vector<cv::Mat> rows;
        for (int r = 0; r < ROW; ++r) {
            std::vector<cv::Mat> slice(frames.begin() + r*COL, frames.begin() + r*COL + COL);
            cv::Mat hr;
            cv::hconcat(slice, hr);
            rows.push_back(hr);
        }
        cv::Mat grid;
        cv::vconcat(rows, grid);

        cv::imshow(win, grid);

        int key = cv::waitKey(1);
        if (key == 'q') break;
        else if (key == ' ') paused = !paused;
        else if (key == 'a' || key == 'A') showNames = !showNames;
        else if (key == 'v' || key == 'V') showProg  = !showProg;
        else if (paused && key == 81) { // left arrow => back 10
            for (int i = 0; i < (int)caps.size(); ++i) {
                int cur = int(caps[i].get(cv::CAP_PROP_POS_FRAMES));
                caps[i].set(cv::CAP_PROP_POS_FRAMES, std::max(cur-10, 0));
            }
        }
        else if (paused && key == 83) { // right arrow => forward 10
            for (int i = 0; i < (int)caps.size(); ++i) {
                int cur = int(caps[i].get(cv::CAP_PROP_POS_FRAMES));
                caps[i].set(cv::CAP_PROP_POS_FRAMES, std::min(cur+10, totalFrames[i]));
            }
        }
    }

    for (auto &c : caps) c.release();
    cv::destroyAllWindows();
    return 0;
}


/*
g++ Projects/MultiVideoPlayer/OpenCV/main.cpp -std=c++17 -o Projects/MultiVideoPlayer/OpenCV/video_grid \
-I/home/d2u25/OCV_GPU/install/include/opencv4 \
-L/home/d2u25/OCV_GPU/install/lib \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_videoio \
-lopencv_imgcodecs \
-lopencv_cudaarithm

Dont run it in VScode

LD_LIBRARY_PATH=/home/d2u25/OCV_GPU/install/lib ./Projects/MultiVideoPlayer/OpenCV/video_grid

*/