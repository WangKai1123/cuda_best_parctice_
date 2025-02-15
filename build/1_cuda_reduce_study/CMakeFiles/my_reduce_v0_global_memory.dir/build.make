# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wk/cuda_best_parctice_

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wk/cuda_best_parctice_/build

# Include any dependencies generated for this target.
include 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/compiler_depend.make

# Include the progress variables for this target.
include 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/progress.make

# Include the compile flags for this target's objects.
include 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/flags.make

1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/flags.make
1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/includes_CUDA.rsp
1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o: /home/wk/cuda_best_parctice_/1_cuda_reduce_study/my_reduce_v0_global_memory.cu
1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wk/cuda_best_parctice_/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o"
	cd /home/wk/cuda_best_parctice_/build/1_cuda_reduce_study && /usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o -MF CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o.d -x cu -c /home/wk/cuda_best_parctice_/1_cuda_reduce_study/my_reduce_v0_global_memory.cu -o CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o

1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target my_reduce_v0_global_memory
my_reduce_v0_global_memory_OBJECTS = \
"CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o"

# External object files for target my_reduce_v0_global_memory
my_reduce_v0_global_memory_EXTERNAL_OBJECTS =

1_cuda_reduce_study/my_reduce_v0_global_memory: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/my_reduce_v0_global_memory.cu.o
1_cuda_reduce_study/my_reduce_v0_global_memory: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/build.make
1_cuda_reduce_study/my_reduce_v0_global_memory: /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart.so
1_cuda_reduce_study/my_reduce_v0_global_memory: /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas.so
1_cuda_reduce_study/my_reduce_v0_global_memory: /usr/lib/x86_64-linux-gnu/librt.so
1_cuda_reduce_study/my_reduce_v0_global_memory: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/linkLibs.rsp
1_cuda_reduce_study/my_reduce_v0_global_memory: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/objects1.rsp
1_cuda_reduce_study/my_reduce_v0_global_memory: 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/wk/cuda_best_parctice_/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable my_reduce_v0_global_memory"
	cd /home/wk/cuda_best_parctice_/build/1_cuda_reduce_study && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_reduce_v0_global_memory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/build: 1_cuda_reduce_study/my_reduce_v0_global_memory
.PHONY : 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/build

1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/clean:
	cd /home/wk/cuda_best_parctice_/build/1_cuda_reduce_study && $(CMAKE_COMMAND) -P CMakeFiles/my_reduce_v0_global_memory.dir/cmake_clean.cmake
.PHONY : 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/clean

1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/depend:
	cd /home/wk/cuda_best_parctice_/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wk/cuda_best_parctice_ /home/wk/cuda_best_parctice_/1_cuda_reduce_study /home/wk/cuda_best_parctice_/build /home/wk/cuda_best_parctice_/build/1_cuda_reduce_study /home/wk/cuda_best_parctice_/build/1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : 1_cuda_reduce_study/CMakeFiles/my_reduce_v0_global_memory.dir/depend

