## Interesting Sources
Interesting opencv doc: [Here](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)



## The Basics

### References and pointers
If you want obvious "no-copy" semantics:

Use pointers (T*) when you want it explicit: “This function takes an address.”

Use references (T&) only when you want to guarantee it can’t be null and can’t change which object it refers to.

For large objects, you can use const T& to avoid copies while still signaling "read-only."

