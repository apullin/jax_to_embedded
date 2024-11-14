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

Apparently, XLA can be configured to dump LLVM IR, even when running via jax via python!

A quick python program uses `jnp` in jax for a simple algebraic function - the quadratic formula.
Once we have LLVM, we can use `clang` to compile

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