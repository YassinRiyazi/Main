/*
 There are some things to note in this code:

    1. The pixels are written out in rows.

    2. Every row of pixels is written out left to right.

    3. These rows are written out from top to bottom.

    4. By convention, each of the red/green/blue components are represented internally by real-valued variables that range from 0.0 to 1.0.
    These must be scaled to integer values between 0 and 255 before we print them out.

    5.Red goes from fully off (black) to fully on (bright red) from left to right, and green goes from fully off at the top (black) to fully on at the bottom (bright green). Adding red and green light together make yellow so we should expect the bottom right corner to be yellow.
*/


// Can use https://github.com/nothings/stb to write images


#ifndef BasicShit
    #define BasicShit
    #include <cmath>
    #include <iostream>
#endif

#include <chrono>
#include <thread>

#include "utils/vec3.h"
#include "utils/color.h"

// using namespace std::chrono_literals;


int main(void){
    int width   = 512;
    int height  = 265;

    
    width = 400;
    double aspect_ratio = 16.0 / 9.0;

    // Calculate the image height, and ensure that it's at least 1.
    height  = int(width / aspect_ratio);
    height  = (height < 1) ? 1 : height;

    // Viewport widths less than one are ok since they are real valued.
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(width)/height);


    // Render
    std::cout << "P3\n" << width << ' ' << height << "\n255\n";
    
    int i = 0, j = 0;
    int r = 0, g = 0, b = 0;
    for(j = 0;j<height;j++){
        std::clog << "\rScanlines remaining: " << (height - j) << ' ' << std::flush;
        for(i = 0;i<width;i++){
            vec3 pixel_color = color(double(i)/(width-1), double(j)/(height-1), 0);
            write_color(std::cout, pixel_color);
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(130));
    }
    return 0;
}