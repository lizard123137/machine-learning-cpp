cmake_minimum_required(VERSION 3.28)
project(machine-learning-cpp LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)

add_executable(main 
    src/main.cpp
    src/perceptron.hpp
    src/neural_layer.hpp
    src/neural_network.hpp
    src/serializer.hpp
)

target_link_libraries(main PRIVATE
    sfml-graphics
    Eigen3::Eigen
)

 target_compile_features(main PRIVATE cxx_std_17)