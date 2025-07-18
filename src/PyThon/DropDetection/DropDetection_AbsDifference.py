import cv2 

class DropPreProcessor:
    """Preprocesses grayscale video frames to detect moving objects (e.g., drops) via frame differencing.
    
    This class supports both CPU and CUDA-accelerated pipelines using OpenCV.
    """

    def __init__(self, kernel_size=(5, 5), threshold_val=30, use_cuda=True):
        """
        Initializes the preprocessor with morphological filters and CUDA setup.

        Args:
            kernel_size (tuple): Size of the structuring element for morphological operations.
            threshold_val (int): Threshold value used to binarize the frame difference.
            use_cuda (bool): Whether to use CUDA acceleration if available.
        """
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.threshold_val = threshold_val

        if self.use_cuda:
            # Initialize CUDA-based morphological filters and GPU memory
            self.morph_open = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, self.kernel)
            self.morph_dilate = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, self.kernel)
            self.gpu_prev = cv2.cuda_GpuMat()
            self.gpu_curr = cv2.cuda_GpuMat()
        else:
            # CUDA not used; filters set to None
            self.morph_open = None
            self.morph_dilate = None

    def process(self, prev_gray, curr_gray):
        """
        Processes a pair of consecutive grayscale frames to extract contours of moving objects.

        Args:
            prev_gray (np.ndarray): Previous grayscale frame.
            curr_gray (np.ndarray): Current grayscale frame.

        Returns:
            list: List of detected contours representing motion between frames.
        """
        if self.use_cuda:
            # Upload frames to GPU memory
            self.gpu_prev.upload(prev_gray)
            self.gpu_curr.upload(curr_gray)

            # Compute absolute difference and threshold on GPU
            diff_gpu = cv2.cuda.absdiff(self.gpu_prev, self.gpu_curr)
            _, thresh_gpu = cv2.cuda.threshold(diff_gpu, self.threshold_val, 255, cv2.THRESH_BINARY)

            # Apply morphological opening and dilation to reduce noise and connect components
            opened_gpu = self.morph_open.apply(thresh_gpu)
            dilated_gpu = self.morph_dilate.apply(opened_gpu)

            # Download the result back to CPU for contour detection
            dilated = dilated_gpu.download()
        else:
            # CPU fallback: frame differencing and morphology
            diff = cv2.absdiff(prev_gray, curr_gray)
            _, thresh = cv2.threshold(diff, self.threshold_val, 255, cv2.THRESH_BINARY)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            dilated = cv2.dilate(opened, self.kernel, iterations=2)

        # Contour detection using CPU
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
