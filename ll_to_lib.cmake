## Convenience function for going from .ll -> linkable cmake library ##
function(add_llvm_ir_target llvm_ir_file target_name)
    # Define paths for the IR file and the compiled object
    set(LLVM_IR_FILE ${llvm_ir_file})
    get_filename_component(LLVM_OBJECT_FILE ${target_name} NAME)
    set(LLVM_OBJECT_FILE ${CMAKE_BINARY_DIR}/${LLVM_OBJECT_FILE}.o)
    set(LIBRARY_NAME "${target_name}_lib")

    # Get a source-relative path for display
    file(RELATIVE_PATH LLVM_IR_FILE_RELATIVE ${CMAKE_SOURCE_DIR} ${LLVM_IR_FILE})

    # Add a custom command to compile the LLVM IR file to an object file
    add_custom_command(
            OUTPUT ${LLVM_OBJECT_FILE}
            COMMAND ${CMAKE_C_COMPILER} -c ${LLVM_IR_FILE} -o ${LLVM_OBJECT_FILE}
            DEPENDS ${LLVM_IR_FILE}
            COMMENT "Compiling LLVM IR ${LLVM_IR_FILE_RELATIVE}"
            VERBATIM
    )

    # Create a static library from the compiled object file
    add_library(${target_name} STATIC ${LLVM_OBJECT_FILE})

    # Set the linker language to C for the library
    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE C)

    # Ensure the library target depends on the custom command output
    add_dependencies(${target_name} gen_llvm_ir)
endfunction()

#function(add_llvm_ir_target llvm_ir_file target_name)
#    # Define paths for the IR file and the compiled object
#    set(LLVM_IR_FILE ${llvm_ir_file})
#    get_filename_component(LLVM_OBJECT_FILE ${target_name} NAME)
#    set(LLVM_OBJECT_FILE ${CMAKE_BINARY_DIR}/${LLVM_OBJECT_FILE}.o)
#    set(LIBRARY_NAME "${target_name}_lib")
#
#    # Get a source-relative path for display
#    file(RELATIVE_PATH LLVM_IR_FILE_RELATIVE ${CMAKE_SOURCE_DIR} ${LLVM_IR_FILE})
#
#    # Add a custom target to compile the LLVM IR file
#    add_custom_target(
#            ${target_name}
#            COMMAND ${CMAKE_C_COMPILER} -c ${LLVM_IR_FILE} -o ${LLVM_OBJECT_FILE}
#            DEPENDS ${LLVM_IR_FILE}
#            BYPRODUCTS ${LLVM_OBJECT_FILE}
#            COMMENT "Compiling LLVM IR ${LLVM_IR_FILE_RELATIVE}"
#    )
#
#    # Create a static library from the compiled object file
#    add_library(${LIBRARY_NAME} STATIC ${LLVM_OBJECT_FILE})
#
#    # Set the linker language to C for the library
#    set_target_properties(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE C)
#
#    # Ensure the library target depends on the compilation target
#    add_dependencies(${LIBRARY_NAME} ${target_name})
#endfunction()