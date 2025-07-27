# C vs. C++

Recently, I have bought the _A Tour of C++_ to deepen my C++ knowledge and become more familiar with C++20. 

I have noticed a condescending tone toward C++ from C developers and vice versa [^1].

[^1]: [Linus Torvalds](https://harmful.cat-v.org/software/c++/linus)

To some degree, I agree that using C++ will introduce overheads, and exception handling can cause problems, but for my work, it is manageable. 
I don't know anything about kernel development and debugging in kernels.

My experience in the HighSpeedCamera project clearly shows that with bare single-thread C, I could get 189 fps, while bare C++ could achieve 122, and with multi-threading, it goes up to 168. With my experience, I agree that for real-time and demanding processes, C is better than C++, but I still personally struggle with appropriate, easy, multi-threaded C code.

[CppCon](https://www.youtube.com/@CppCon)

[meetingcpp](https://meetingcpp.com/)