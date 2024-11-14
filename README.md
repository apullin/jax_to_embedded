# Jax-to-embedded Prototype

This is a workup of an idea I had, and as a skill-builder to learn something about jax.

### The idea

Can I use jax to generate a program that can ultimately be compiled to run on an embedded
microcontroller?

This is just a proof of concept.  
Eventually, in the very long run, I would like to do something clever like
[this jit dot product](https://github.com/dddrrreee/cs240lx-24spr/tree/main/labs/1-dynamic-code-gen#part-5-make-a-jitter-for-dot-product).  
That exercise was particularly illuminating, as it generated a program with the data "baked" into it,
rather than just converting to another library call.

(for example: Could we turn an MLP into a purely in-order instruction stream, with no separate memory loads?)

### Method

Apparently, XLA can be configured to dump LLVM IR, even when running via jax via python.

A quick python program uses `jnp` in jax for a simple algebraic function - the quadratic formula.
Once we have LLVM, we can use `clang` to compile

So we will go:  
python --> jax --> HLO --> LLVM IR -> compiled object --> link to C code

### Details

A bunch of code and intermediates are listed here, for easy viewing.

This is the jax function we are going to jit:
```
# Function that we will JIT (quadratic roots)
def quad_roots(a : float ,b : float ,c : float  ):
    return ( (-b + jnp.sqrt( b**2 - 4*a*c ))/(2*a), (-b - jnp.sqrt( b**2 - 4*a*c ))/(2*a) )
```

Which generates the HLO:
```
HloModule jit_quad_roots, entry_computation_layout={(f32[], f32[], f32[])->(f32[], f32[])}, allow_spmd_sharding_propagation_to_output={true,true}

ENTRY main.25 {
  Arg_1.2 = f32[] parameter(1), sharding={replicated}
  negate.6 = f32[] negate(Arg_1.2), metadata={op_name="jit(quad_roots)/jit(main)/neg" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  multiply.7 = f32[] multiply(Arg_1.2, Arg_1.2), metadata={op_name="jit(quad_roots)/jit(main)/integer_pow[y=2]" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  Arg_0.1 = f32[] parameter(0), sharding={replicated}
  constant.5 = f32[] constant(4)
  multiply.8 = f32[] multiply(Arg_0.1, constant.5), metadata={op_name="jit(quad_roots)/jit(main)/mul" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  Arg_2.3 = f32[] parameter(2), sharding={replicated}
  multiply.9 = f32[] multiply(multiply.8, Arg_2.3), metadata={op_name="jit(quad_roots)/jit(main)/mul" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  subtract.10 = f32[] subtract(multiply.7, multiply.9), metadata={op_name="jit(quad_roots)/jit(main)/sub" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  sqrt.11 = f32[] sqrt(subtract.10), metadata={op_name="jit(quad_roots)/jit(main)/sqrt" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  add.12 = f32[] add(negate.6, sqrt.11), metadata={op_name="jit(quad_roots)/jit(main)/add" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  constant.4 = f32[] constant(2)
  multiply.13 = f32[] multiply(Arg_0.1, constant.4), metadata={op_name="jit(quad_roots)/jit(main)/mul" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  divide.14 = f32[] divide(add.12, multiply.13), metadata={op_name="jit(quad_roots)/jit(main)/div" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  negate.15 = f32[] negate(Arg_1.2), metadata={op_name="jit(quad_roots)/jit(main)/neg" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  multiply.16 = f32[] multiply(Arg_1.2, Arg_1.2), metadata={op_name="jit(quad_roots)/jit(main)/integer_pow[y=2]" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  multiply.17 = f32[] multiply(Arg_0.1, constant.5), metadata={op_name="jit(quad_roots)/jit(main)/mul" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  multiply.18 = f32[] multiply(multiply.17, Arg_2.3), metadata={op_name="jit(quad_roots)/jit(main)/mul" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  subtract.19 = f32[] subtract(multiply.16, multiply.18), metadata={op_name="jit(quad_roots)/jit(main)/sub" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  sqrt.20 = f32[] sqrt(subtract.19), metadata={op_name="jit(quad_roots)/jit(main)/sqrt" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  subtract.21 = f32[] subtract(negate.15, sqrt.20), metadata={op_name="jit(quad_roots)/jit(main)/sub" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  multiply.22 = f32[] multiply(Arg_0.1, constant.4), metadata={op_name="jit(quad_roots)/jit(main)/mul" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  divide.23 = f32[] divide(subtract.21, multiply.22), metadata={op_name="jit(quad_roots)/jit(main)/div" source_file="/Users/andrewpullin/personal/jax_to_embedded/jax_test.py" source_line=11}
  ROOT tuple.24 = (f32[], f32[]) tuple(divide.14, divide.23)
} // main.25
```

Which generates the optimized LLVM IR ( in `xla_dump/module_0000.jit_quad_roots.ir-with-opt.ll`):
```
; ModuleID = '__compute_module'
source_filename = "__compute_module"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin24.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define void @main.25(ptr nocapture readnone %retval, ptr noalias nocapture readnone %run_options, ptr noalias nocapture readnone %params, ptr noalias nocapture readonly %buffer_table, ptr noalias nocapture readnone %status, ptr noalias nocapture readnone %prof_counters) local_unnamed_addr #0 {
entry:
  %0 = getelementptr inbounds ptr, ptr %buffer_table, i64 3
  %Arg_0.1 = load ptr, ptr %0, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %1 = getelementptr inbounds ptr, ptr %buffer_table, i64 5
  %Arg_1.2 = load ptr, ptr %1, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %2 = getelementptr inbounds ptr, ptr %buffer_table, i64 4
  %Arg_2.3 = load ptr, ptr %2, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %3 = getelementptr inbounds ptr, ptr %buffer_table, i64 1
  %fusion.2 = load ptr, ptr %3, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %4 = load float, ptr %Arg_1.2, align 4, !invariant.load !0, !noalias !2
  %multiply.4 = fmul float %4, %4
  %5 = load float, ptr %Arg_0.1, align 4, !invariant.load !0, !noalias !2
  %multiply.3 = fmul float %5, 4.000000e+00
  %6 = load float, ptr %Arg_2.3, align 4, !invariant.load !0, !noalias !6
  %multiply.2 = fmul float %multiply.3, %6
  %subtract.1 = fsub float %multiply.4, %multiply.2
  %7 = tail call float @llvm.sqrt.f32(float %subtract.1)
  %8 = getelementptr inbounds ptr, ptr %buffer_table, i64 2
  %fusion = load ptr, ptr %8, align 8, !invariant.load !0, !dereferenceable !1, !align !1
  %9 = fneg float %4
  %subtract.0 = fsub float %9, %7
  %multiply.0 = fmul float %5, 2.000000e+00
  %divide.0 = fdiv float %subtract.0, %multiply.0
  store float %divide.0, ptr %fusion, align 4, !alias.scope !7, !noalias !8
  %add.0 = fsub float %7, %4
  %divide.1 = fdiv float %add.0, %multiply.0
  store float %divide.1, ptr %fusion.2, align 4, !alias.scope !6, !noalias !10
  %tuple.24 = load ptr, ptr %buffer_table, align 8, !invariant.load !0, !dereferenceable !11, !align !11
  store ptr %fusion.2, ptr %tuple.24, align 16, !alias.scope !12, !noalias !2
  %10 = getelementptr inbounds [2 x ptr], ptr %tuple.24, i64 0, i64 1
  store ptr %fusion, ptr %10, align 8, !alias.scope !12, !noalias !2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "denormal-fp-math"="preserve-sign" "no-frame-pointer-elim"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{}
!1 = !{i64 4}
!2 = !{!3, !5}
!3 = !{!"buffer: {index:1, offset:0, size:4}", !4}
!4 = !{!"XLA global AA domain"}
!5 = !{!"buffer: {index:2, offset:0, size:4}", !4}
!6 = !{!3}
!7 = !{!5}
!8 = !{!9, !3}
!9 = !{!"buffer: {index:0, offset:0, size:16}", !4}
!10 = !{!9, !5}
!11 = !{i64 16}
!12 = !{!9}
```

In this case, there are no external dependencies, other than `llvm.sqrt.f32`. So we should be able to 
build this for our Cortex-M target!

TODO: If I build with `clang`, will it pull in a clang `sqrt` function, and not use the libm impl?

## Instructions - Host

The `runner` dir provides a project that compiles the llvm IR and provides a header, and a `main.c`
to run an example.

For convenience, and to deal with the necessary input format to the generated function, a wrapper
function in C that sets up the "buffer table" is also provided.

The cmake build process will run the python to make the XLA dump, then compile that in.

Install python by any means.  
Install jax: `pip install jax`

Build and run via cmake:
```
cmake -S . -B build && cmake --build build --target runner
./build/runner/runner
```

## Instructions - Embedded

Sadly, this does not currently work. But I think it is just the entry/startup for qemu.

Following embedded tradition, "host" means PC, and "target" means the non-pc device, in this case, a generic Cortex-M4
microcontroller in a "bare metal" environment.

Notably: This uses the same `main.c` and header that the host-side does!

BUT:  
- the compile and link works
- looking at the `objdump -S module_0000.jit_quad_roots.ir-with-opt.o`, it looks like FPU instructions are emitted

You will need `arm-none-eabi-gcc` on your `PATH`.

CMSIS-5 is a submodule (it's large, since it include their whole NN and DSP library).

```
git submodule update --init --recursive
cd cm4-qemu
cmake -S . -B build && cmake --build build --target qemu
./build/qemu/qemu
```

#### Don't other MLIR frameworks for embedded exist?

Sure. tinyTVM, tinyIREE. I suppose I'll have to learn them all.

#### Doesn't tool X already exist to do this much more directly?

Maybe. Let me know.

## Quirks

- The "buffer table" input to generated function is all done by discovery - it seems to work, 
but maybe there's a more formal way to construct it.
- The order of arguments appears to be arbitary - TODO can it be controlled in jax or XLA?
- The function name in the generated `.ll` is random, apparently - TODO edit with `sed`?

## Future

A simple matrix dot product wrapper would be the next steps. afaik, this will just target `eigen`,
which I could then shim to the CMSIS-DSP library.