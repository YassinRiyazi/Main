
void swap_c(int *a, int *b) {
    /*
    * @Simple C function to swap two integers. Use to benchmark against C++. And a excuse
    * to practice C and GDB.
    * @param int *a Pointer to the first integer to swap.
    * @param int *b Pointer to the second integer to swap.
    * @return void
    */
    int temp = *a;
    *a = *b;
    *b = temp;
}