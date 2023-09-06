# QALIBOO
QALIBOO is a fork of Cornell-MOE (Cornell-Metrics-Optimization-Engine) that allows you to run 
the parallel Knowledge Gradient optimization algorithm (q-KG) on finite domains, namely precomputed functions.

> *NOTE*: this documentation is largely based on the original `README.MD`
> Please refer to that file for more details

## Step-by-Step Installation
We only tested an installation based on Anaconda/Miniconda 
on a Ubuntu 20.04 machine. Other Linux distributions should be OK too.

To install on an Apple machine, please refer to the original `README.MD`

> **IMPORTANT**: As far as we know, it is **NOT** possible to build
> the code on newer Apple machines with M1 processors.

> **IMPORTANT**: There are no installation instructions for Windows.

#### 0.1 Clone the repository
```bash
$ git clone https://github.com/Vysybyl/QALIBOO
$ cd QALIBOO
```

#### 0.2 Install conda
Install [Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
and initialize conda

#### 1. Create conda environment
Create and activate a new `conda` environment
```bash
$ conda env create -f qaliboo_env.yaml --name qaliboo
$ conda activate qaliboo
```

#### 1-bis Additional build requirements
> You can skip this step for now. Come back here if the build fails

Many build tools are required to compile the code, namely `gcc 4.7.2+`, `cmake 2.8.9+`, `boost 1.51+`, etc.
If the build fails, one or more of them may be missing from your machine
(check the error message!).

In that case, install them

```bash
$ sudo apt-get update
$ sudo apt-get install python3-dev gcc cmake libboost-all-dev libblas-dev liblapack-dev gfortran
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


## Running QALIBOO

The program is shipped with three test precomputed functions / datasets:
* LiGen
* Query26
* StereoMatch

To run the program with default parameters, just use
```bash
$ python main.py -p <function-name>
```
For example
```bash
$ python main.py -p Query26
```

> NOTE: In some cases, if the internal q-EI does not converge, you may see some
logs warning that a fallback algorithm it is used. This is not an error

The results are saved in JSON format under `results/simplified_runs`

For more information on other parameters, run
```bash
$ python main.py --help
```
