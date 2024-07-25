# PAKMAN
PAKMAN (PArallel Knowledge with MAchiNe learning) is an optimization library that integrates parallel Bayesian Optimization (BO) with machine learning (ML) models. This combination is designed to efficiently tackle complex optimization problems, especially those involving constraints and high-dimensional spaces. PAKMAN leverages new acquisition functions and both synchronous and asynchronous execution strategies to achieve faster convergence and superior performance.

## Step-by-Step Installation with Conda virtual environment
We only tested an installation based on Anaconda/Miniconda 
on a Ubuntu 20.04 machine.

#### 0.1 Clone the repository
```bash
$ git clone https://github.com/baroneleonardo/PAKMAN
$ cd PAKMAN
```

#### 0.2 Install conda
Install [Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
and initialize conda

#### 1. Create conda environment
Create and activate a new `conda` environment
```bash
$ conda env create -f qaliboo_env.yaml --name qaliboo
$ conda activate qaliboo
$ conda install -c conda-forge boost=1.74 python=3.8
```

#### 1-bis Additional build requirements
> You can skip this step for now. Come back here if the build fails

Many build tools are required to compile the code, namely `gcc 4.7.2+`, `cmake 2.8.9+`, `boost 1.51+`, etc.
If the build fails, one or more of them may be missing from your machine
(check the error message!).

In that case, install them

```bash
$ sudo apt-get update
$ sudo apt-get install python3-dev gcc cmake libboost-all-dev libblas-dev g++ liblapack-dev gfortran
```

#### 3. Set environment variables
To set the correct environment variables for compiling the C++ code. 
One need to create a script with the content as follows, then **_source_** it.
```bash
export MOE_CC_PATH=/path/to/your/gcc && export MOE_CXX_PATH=/path/to/your/g++
export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/path/to/where/Python.h/is/found -D MOE_PYTHON_LIBRARY=/path/to/python/shared/library/object"
```
For example, the script on a AWS EC2 with Ubuntu OS is as follows
```bash
#!/bin/bash

export MOE_CC_PATH=/usr/bin/gcc
export MOE_CXX_PATH=/usr/bin/g++

export MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/usr/include/python2.7 -D MOE_PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0"
```

#### 4. Build
The program is build with
```bash
$ python setup.py build_ext
```

If you see any issue, you may want to use the
following script that captures only the errors from the 
build logs
```bash
$ chmod +x ./build_and_check_for_errors.sh
$ ./build_and_check_for_errors.sh
```

#### 5. (Optional) Install
If you wish to use the package from other locations
then the root folder of the project, install it in
development mode:
```bash
$ python -m pip install -e .
```
## Installation with Docker
If you want to install the library using Docker, download the dockerfile.

## Running PAKMAN
