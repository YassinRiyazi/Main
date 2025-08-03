## Notes on C
- Use `stdio.h` for basic I/O operations.
- Pointers can be tricky; always double-check dereferencing.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.
> ## Abusing C 😅🫠
> ### Calling the Main function recursively as a system reboot
> Technically allowed by the standard, but there’s no guarantee of stack cleanup or consistent behavior.
>
> Why it’s bad: You’re consuming stack space with each call. Most systems will eventually crash with a stack overflow.
> ```C
>     #include <stdio.h>
> 
>     int main(void) {
>         static int count = 0;
>         printf("Restart #%d\n", count++);
>         if (count < 3) main(); // risky "reboot"
>         return 0;
>         }
> ```
> ---
> ### Modifying string literals
> Why it isn’t good: String literals are typically stored in read-only memory. Writing to them may crash or silently corrupt memory.
> ```C
>     char *s = "hello";
>     s[0] = 'H'; // Undefined behavior
> ```
> ---
> ### Using void main()
> Why it’s bad: The C standard specifies that main should return int.
> Using void main() is non-standard and can lead to unexpected behavior on some systems where the return value is expected by the operating system or runtime environment.
> ```C
> void main() {
>     printf("Hello, world!\n");
> }
> ```
> ---
> ### Flushing input streams with __*fflush*__
> 
> Why it’s weird: __*fflush*__ is only defined for output streams in standard C.
> Using it on input streams like stdin results in undefined behavior.
> 
> What happens: The outcome is unpredictable and varies between compilers.
> It might appear to work on some systems but fail or cause issues on others.
> ```C
> #include <stdio.h>
> 
> int main() {
>     int x;
>     printf("Enter a number: ");
>     fflush(stdin); // Undefined behavior
>     scanf("%d", &x);
>     return 0;
> }
> ```
> ---
> ### Macros with side effects
> Why it’s weird: Macros are simple text substitutions, so including expressions with side effects (like increments) can lead to multiple evaluations and undefined behavior.
> 
> What happens: The side effects are applied each time the argument appears in the macro expansion, potentially causing unexpected results.
> ```C
> #include <stdio.h>
> 
> #define SQUARE(x) ((x) * (x))
> 
> int main() {
>     int i = 5;
>     int result = SQUARE(i++); // Expands to ((i++) * (i++)), undefined behavior
>     printf("Result: %d, i: %d\n", result, i);
>     return 0;
> }
> ```



---
---
---




> [!TIP]
> ## Weird C behavior
> ### Value modification in one line (undefined behavior)
> Value modification in one line. It is heavily dependent on the optimization level and compiler.
>
> What happens: The order of evaluation of function arguments is unspecified.
> Combined with post-increment side effects, this means output varies depending on compiler and optimization.
>```C
>     #include <stdio.h>
> 
>     int main(){
>         int i=0;
> 
>         printf("%d %d %d",i,i++,i++);
>         return 0;
>     }
>```
> ---
> ### Signed integer overflow
> Why it’s weird: Unsigned overflow is well-defined (wraparound), but signed overflow is undefined, allowing compilers to make unexpected optimizations.
> 
> ```C
> int x = 2147483647;  // INT_MAX on 32-bit
> x++;
> printf("%d\n", x);   // Undefined behavior
> ```
> ---
> ### Accessing freed memory
> What happens: May print 42, crash, or behave differently on each run.
> ```C
> int *p = malloc(sizeof(int));
> *p = 42;
> free(p);
> printf("%d\n", *p); // Undefined behavior
> ```
> ---
> ### Using uninitialized variables
> Why it’s bad: Uninitialized variables contain indeterminate values, which can lead to unpredictable behavior or bugs that are hard to trace.
> 
> What happens: The value of an uninitialized variable is whatever was in that memory location, which could be anything, leading to inconsistent results.
> 
> ```C
> #include <stdio.h>
> 
> int main() {
>     int x; // Uninitialized
>     printf("%d\n", x); // Undefined behavior
>     return 0;
> }
> ```
> ---
> ### Ignoring return values of critical functions
> Why it’s bad: Functions like __*malloc*__ or __*fopen*__ can fail and return NULL. Ignoring these return values can lead to dereferencing null pointers or other errors.
> 
> What happens: If you don’t check for failure, your program might crash or behave unexpectedly when resources are unavailable.
> ```C
> #include <stdio.h>
> #include <stdlib.h>
> 
> int main() {
>     int *p = malloc(sizeof(int)); // No check for NULL
>     *p = 42; // Undefined behavior if malloc failed
>     printf("%d\n", *p);
>     free(p);
>     return 0;
> }
> ```
> ---
> ### Direct comparison of floating-point numbers
> Why it’s weird: Due to floating-point precision limitations, two numbers that are mathematically equal might not compare as equal in C.
> 
> What happens: Direct comparisons like == can fail unexpectedly, even when the values should logically be the same.
> ```C
> #include <stdio.h>
> 
> int main() {
>     float a = 0.1;
>     float b = 0.2;
>     float c = a + b;
>     if (c == 0.3) {
>         printf("Equal\n");
>     } else {
>         printf("Not equal\n"); // This might be printed
>     }
>     return 0;
> }
> ```