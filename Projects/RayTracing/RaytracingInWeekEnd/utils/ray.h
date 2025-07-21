#include "vec3.h"

#ifndef RAY_H
    #define RAY_H

    class ray {
        /*
            $$\mathbf{P}(t) = \mathbf{A} + t \mathbf{b}$$
            represent the function P(t) as a function that we'll call ray::at(t): 
        */
        private:
            point3 orig;
            vec3 dir;
        

        public:
            ray() {}
            
            ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}
            
            
            point3 at(double t) const {
                return orig + t*dir;
            }
            /*
                the functions ray::origin() and ray::direction() both return an immutable reference to their members.
                Callers can either just use the reference directly, or make a mutable copy depending on their needs.)

            */
            const point3& origin() const  { return orig; }
            const vec3& direction() const { return dir; }
        }


#endif