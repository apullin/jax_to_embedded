#include <stdio.h>
#include <math.h>

#pragma clang optimize off

#include "jax_quad_roots.h"

// Manually written wrapper, to do all the in/out reshaping
void quad_roots(float inputs[3], float outputs[2])
{
    // Set up buffer table for llvm function signature
    void* buffer_table[6];
    void* tuple[2] = { &outputs[0], &outputs[1] };

    // Set up the buffer table
    buffer_table[0] = tuple;        // Tuple of output pointers
    buffer_table[1] = &outputs[0];  // Output buffer for first result
    buffer_table[2] = &outputs[1];  // Output buffer for second result
    // Note here that args order is seemingly arbitrary, and the only way to figure it out
    //  is to observe the HLO text or IR code itself...
    buffer_table[3] = &inputs[0];   // Input a (float)
    buffer_table[4] = &inputs[2];   // Input c (float)
    buffer_table[5] = &inputs[1];   // Input b (float)

    // Call the function in the generated llvm ir
    main_25(NULL, NULL, NULL, buffer_table, NULL, NULL);
}

int main(void) {
    // Inputs and Outputs
    float inputs[3] = {2.0f, 0.0f, -32.0f};
    float outputs[2] = {NAN, NAN};

    // Call the convenient wrapper function
    quad_roots(inputs, outputs);

    printf("result : (%f, %f)\n", outputs[0], outputs[1]);

    return 0;
}
