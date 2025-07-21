#ifndef BasicShit
    #define BasicShit
    #include <cmath>
    #include <iostream>
#endif

#ifndef VEC3_H
    #define VEC3_H

    class vec3 {
        public:
            //Stores the 3D vector components x, y, z in an array e[0], e[1], e[2].
            double e[3];

            // Default constructor:         initializes to (0,  0,  0).
            // Parameterized constructor:   initializes to (e0, e1, e2)
            vec3() : e{0,0,0} {}
            vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

            // Provide readable access to components: x(), y(), z().
            double x() const { return e[0]; }
            double y() const { return e[1]; }
            double z() const { return e[2]; }

            // Negates the vector: -v.
            vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
            // Allows access like v[0] or v[i], both for reading and writing
            double operator[](int i) const { return e[i]; }
            double& operator[](int i) { return e[i]; }

            // Adds another vector v to the current vector.
            vec3& operator+=(const vec3& v) {
                e[0] += v.e[0];
                e[1] += v.e[1];
                e[2] += v.e[2];
                return *this;
            }

            //Scales the vector by t or divides by t (optimized by multiplying by reciprocal).
            vec3& operator*=(double t) {
                e[0] *= t;
                e[1] *= t;
                e[2] *= t;
                return *this;
            }

            vec3& operator/=(double t) {return *this *= 1/t;}
            
            // length() computes the Euclidean norm (vector magnitude).
            double length() const {return std::sqrt(length_squared());}
            double length_squared() const {return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];}
    };
    // point3 is just an alias for vec3, but useful for geometric clarity in the code.
    using point3 = vec3;
#endif