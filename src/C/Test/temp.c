/**
 * @brief Calculates the factorial of a number.
 * 
 * @param n The number to calculate factorial for.
 * @return unsigned long The factorial of n.
 * 
 * @note Returns 1 if n is 0.
 */
#include <stdio.h>

unsigned long factorial(unsigned int n) {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

/*
 *
 * @brief Reverses a string in place.
 * <pre><code class="language-c">
 * #include <stdio.h> \n
 * 
 * int main() {
 *     printf("Hello, world!\n");
 *     return 0;
 * }
 * </code></pre>
 * 
 * @param str The string to reverse.
 * @return void
 * 
 * @warning The input string must be null-terminated.
 */
void reverse_string(char *str) {
    if (str == NULL) return;
    
    int length = strlen(str);
    for (int i = 0; i < length / 2; i++) {
        char temp = str[i];
        str[i] = str[length - i - 1];
        str[length - i - 1] = temp;
    }
}