# Notes on Python
- Using \__init__.py will fail in docs. 

- Adding path of modules:
    ```Python
    import sys
    import os

    # Add the absolute path to the ./src folder
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../', 'src/PyThon/Utils/Performance')))

    # Now you can import normally
    from Performance_mesurements import average_performance,measure_performance
    ```

### Documantation
    To provide type checking of multiple types use __*Union*__
    ```Python
    from typing import Union, Optional, Tuple
    ```

### Interesting
interesting opencv doc: [Here](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)


TODO:
    Read [multiprocessing — Process-based parallelism](https://docs.python.org/3/library/multiprocessing.html)