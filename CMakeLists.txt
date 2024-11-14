cmake_minimum_required(VERSION 3.28)
project(jax_to_embedded C)

set(CMAKE_C_STANDARD 11)

set( XLA_DUMP_DIR ${CMAKE_SOURCE_DIR}/xla_dump )
set( XLA_FLAGS "--xla_dump_to=${XLA_DUMP_DIR} --xla_dump_hlo_as_text --xla_dump_hlo_as_dot" )

add_custom_command( #gen_llvm_ir
        OUTPUT ${CMAKE_SOURCE_DIR}/xla_dump
        COMMAND env "XLA_FLAGS=${XLA_FLAGS}" python jax_test.py
        DEPENDS jax_test.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running python to generate LLVM IR dump..."
        VERBATIM
)

add_custom_target(gen_llvm_ir DEPENDS ${CMAKE_SOURCE_DIR}/xla_dump/)

include( ll_to_lib.cmake )

# Add a "runner" project to build llvm and a main to run it, as an example
add_subdirectory( runner )