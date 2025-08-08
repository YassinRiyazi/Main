Re-implement 4S-FROF in C++:

Q2 (months 4–6) — Foundations II (C++ concurrency & tooling)
    Topics: Modern C++ features 
                                RAII,
                                move semantics,
                                smart pointers,
                                templates,
                        multithreading
                                Pthreads basics,
                                C++ threads,
                                std::async,
                        unit tests,
                        sanitizers

    Tools: valgrind/ASAN/TSAN, clang-tidy, gtest.
    Book: Effective C++ + C++ Concurrency in Action.
    Project: Implement a producer/consumer camera simulator with a thread for capture and a thread for processing; debug and fix race conditions.

Milestone: Threaded pipeline that ingests synthetic frames at target rate and processes with no data races; documented profiling run.
Q3 (months 7–9) — Numerical methods & inverse problems intro
    Topics: PDE basics, discretization, Tikhonov regularization, adjoint concept introduction.
    Tools: Implement small solvers (conjugate gradients), use numpy/scipy.
    Project: Solve a simple inverse problem (1D/2D Poisson inverse) and demonstrate regularization and parameter selection (L-curve).
    Reading: Nocedal & Wright (selected chapters); papers on inverse problems.

Milestone: Notebook + report showing stable reconstructions and hyperparameter studies.
Q4 (months 10–12) — Computer vision & optics foundations
    Topics: Camera models, calibration, photogrammetry basics, reflection models (Lambertian, specular), photometric stereo basics.
    Tools: OpenCV, COLMAP (learn to run an SfM tool).
    Project: Calibrate camera(s) and reconstruct a simple scene using multi-view / photometric methods; quantify re-projection error.
    Milestone:** Working pipeline that takes video frames → calibrated intrinsics → sparse 3D reconstruction.**




C++:    
    CMake,
    googletest,
    clang-tidy,
    valgrind,
    gdb,
    git,
    AddressSanitizer/UndefinedBehaviorSanitizer,
    pybind11.

Learn the basics of GDB and ASAN
    Reproduce a small memory bug and fix it.

Concrete micro-projects (first 6 months)
    Q1 micro-project: C++ image loader + simple difference-based drop detector. CI + unit tests + benchmark vs Python.
    Q2 micro-project: Threaded capture→process pipeline with race detection fixed and sanitizers enabled.
    Q3 micro-project: Small inverse Poisson problem solved with CG and Tikhonov; plot L-curve and discuss regularization.
    Q4 micro-project: Calibrate camera & compute reprojection error on a small dataset; produce rendered overlay.

Weekly / monthly cadence & study time
    Weekly: 50-70 hrs study + project work. 
    
    Split:  40% reading/lectures,
            50% hands-on coding/experiments,
            10% writing/documentation.

    End of each quarter: deliver a short report (2–4 pages) and a runnable artifact (repo, jupyter notebook, binary) showing results and reproducing key numbers.


## Missing skills / gaps I see (explicit)
Profiling & performance engineering (NVidia Nsight, Intel VTune, perf) — necessary to know where to optimize.
Advanced data structures/algorithms and numerical linear algebra (beyond basics) — for scalable solvers.
Code hygiene & static analysis (clang-tidy, ASAN/UBSAN/TSAN) — avoids subtle bugs in multithreaded code.
Finite-volume/finite-element CFD basics — useful to understand fluids when you need physical priors.



