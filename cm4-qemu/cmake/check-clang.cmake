# Check Clang version manually
execute_process(
        COMMAND ${CLANG_PATH} --version
        OUTPUT_VARIABLE CLANG_VERSION_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract the major version (assumes output like 'clang version 16.0.0')
string(REGEX MATCH "([0-9]+)\\." CLANG_MAJOR_VERSION "${CLANG_VERSION_OUTPUT}")

# Enforce the version
if (CLANG_MAJOR_VERSION LESS 16)
    message(FATAL_ERROR "Clang 16 or higher is required. Found version ${CLANG_MAJOR_VERSION}")
endif ()