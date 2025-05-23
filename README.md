cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit. 
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

# How to build and run lesson code snippet

Here is an example for Lesson 2 Code Snippet 

nvcc -lineinfo hello_blockIdx.cu -o example1.exe

./example1.exe

# Command for each problem set

PS1 
```
./../bin/HW1 /home/zhengwang/repositories/intro_to_parallel_programming/Problem_Sets/Problem_Set_1/cinque_terre_small.jpg test_HW1.jpg
./../bin/HW2 /home/zhengwang/repositories/intro_to_parallel_programming/Problem_Sets/Problem_Set_2/cinque_terre_small.jpg test_hw2.jpg
```

