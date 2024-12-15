cmake_minimum_required(VERSION 3.28)

# This include will set up the toolchain for the embedded target; arm-gcc will be found from your system path
include(cmake/arm-gcc.cmake)

project(cm4_qemu C ASM)

set(CMAKE_C_STANDARD 11)

# You'll need to download and unpack clang > 16,
# e.g. https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.8
set(CLANG_PATH ~/clang+llvm-18.1.8-arm64-apple-macos11/bin/clang )

get_filename_component(CLANG_PATH_ABS "${CLANG_PATH}" ABSOLUTE)

# Check for clang > 16 , needed now that XLA generated IR using opaque pointers.
# Notably, OSX only includes clang v14
include( cmake/check-clang.cmake )

## Compile llvm ir file to an object, targeting cortex-m4; requires clang to be on PATH
set( XLA_DUMP_DIR ${CMAKE_SOURCE_DIR}/../xla_dump )
set( LLVM_IR_FILE ${XLA_DUMP_DIR}/module_0000.jit_quad_roots.ir-with-opt.ll )
set( OBJ_FILE ${CMAKE_BINARY_DIR}/module_0000.jit_quad_roots.ir-with-opt.o )
set( CLANG_OPTS -target arm-none-eabi -mcpu=cortex-m4 -mthumb -opaque-pointers )
add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${CLANG_PATH_ABS} ${CLANG_OPTS} -c -o ${OBJ_FILE} ${LLVM_IR_FILE}
        DEPENDS ${LLVM_IR_FILE}
        COMMENT "Compiling ${LLVM_IR_FILE} to ${OBJ_FILE}"
)

add_library( jax_quad_roots_lib STATIC ${OBJ_FILE} )
set_target_properties( jax_quad_roots_lib PROPERTIES LINKER_LANGUAGE C )


add_executable( cm4_runner
        boot.s
        system_ARMCM4.c
        ../runner/main.c
        ../runner/jax_quad_roots.h
)

target_compile_options( cm4_runner PUBLIC -DARMCM4_FP -D_GNU_SOURCE )

target_link_options( cm4_runner PUBLIC -lc -lgcc -lrdimon )
#target_link_options( cm4_qemu PUBLIC CMSIS5 )
target_link_options( cm4_runner PRIVATE -T "${CMAKE_SOURCE_DIR}/link.ld" )
target_link_options( cm4_runner PRIVATE "-Wl,-Map=${CMAKE_PROJECT_NAME}.map" )

# Link the clang-compiled obj to our exe
target_link_libraries( cm4_runner PRIVATE jax_quad_roots_lib )

add_custom_target( qemu
    COMMAND qemu-system-arm -machine lm3s6965evb -cpu cortex-m4 -m 128k
        -nographic -serial null -monitor null -semihosting
        -kernel $<TARGET_FILE:cm4_runner>
)
add_dependencies( qemu cm4_runner )

add_library( CMSIS5 INTERFACE )
target_include_directories( CMSIS5 INTERFACE
        CMSIS_5/
        CMSIS_5/CMSIS/Core/Include
        CMSIS_5/Device/ARM/ARMCM4/Include/
)
target_link_libraries( cm4_runner PUBLIC CMSIS5 )