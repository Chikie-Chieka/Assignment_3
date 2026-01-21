cmake_minimum_required(VERSION 3.16)
project(experiment_final_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenSSL REQUIRED)

find_package(LibOQS QUIET)
if (LibOQS_FOUND)
  message(STATUS "Found liboqs via CMake")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OQS REQUIRED IMPORTED_TARGET liboqs)
endif()

add_executable(experiment_final
  src/main.c
  src/util.c
  src/csv.c
  src/models.c
  src/model_oqs_kem.c
  src/model_x25519.c
  src/model_ascon80pq.c
  src/ent_phase.c
)

target_include_directories(experiment_final PRIVATE include)
target_link_libraries(experiment_final PRIVATE OpenSSL::Crypto pthread)

if (LibOQS_FOUND)
  target_link_libraries(experiment_final PRIVATE oqs)
else()
  target_link_libraries(experiment_final PRIVATE PkgConfig::OQS)
endif()