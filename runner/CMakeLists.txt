cmake_minimum_required(VERSION 3.28)
project(runner C)

set(CMAKE_C_STANDARD 23)

# Define the LLVM IR file
set( LLVM_IR_FILE ${XLA_DUMP_DIR}/module_0000.jit_quad_roots.ir-with-opt.ll )

# Compile .ll to a static library
add_llvm_ir_target (${LLVM_IR_FILE} jax_quad_roots_lib )

add_executable(runner
        main.c
        jax_quad_roots.h
)

# Link to the compiled llvm lib
target_link_libraries(runner PRIVATE jax_quad_roots_lib)
