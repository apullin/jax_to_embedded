#pragma once

// Inconsistency in naming between clang and gcc
#if defined(__clang__)
#define ASM_FUNC  "_main.25"
#elif defined( __GNUC__ )
#define ASM_FUNC  "main.25"
#endif

// For LLVM-generated function

void main_25(void* retval, void* run_options, void* params, void* buffer_table[], void* status, void* prof_counters) asm(ASM_FUNC);

// Wrapper, manually written
void jax_quad_roots( float args[3], float result[2] );