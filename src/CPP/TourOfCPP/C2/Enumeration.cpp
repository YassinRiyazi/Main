#include <iostream>
enum class TrafficLight {Green, Yellow, Red};

TrafficLight operator++(TrafficLight& light) {
    switch (light) {
        case TrafficLight::Green:
            light = TrafficLight::Yellow;
            break;
        case TrafficLight::Yellow:
            light = TrafficLight::Red;
            break;
        case TrafficLight::Red:
            light = TrafficLight::Green;
            break;
    }
    return light;
}

int main() {
    TrafficLight light = TrafficLight::Red;
    if (light == TrafficLight::Red) {
        std::cout << "Traffic light is Red" << std::endl;
    }
    return 0;
}