
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python3-dev \
    gcc \
    g++ \
    cmake \
    libboost-all-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    wget \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -c conda-forge boost=1.74 python=3.8

RUN git clone https://github.com/baroneleonardo/QALIBOO /QALIBOO
WORKDIR /QALIBOO

COPY qaliboo_env.yaml .
RUN conda env create -f qaliboo_env.yaml --name qaliboo
RUN echo "source activate qaliboo" >> ~/.bashrc
ENV PATH /opt/conda/envs/qaliboo/bin:$PATH

ENV MOE_CC_PATH=/usr/bin/gcc
ENV MOE_CXX_PATH=/usr/bin/g++
ENV MOE_CMAKE_OPTS="-D MOE_PYTHON_INCLUDE_DIR=/opt/conda/envs/qaliboo/include/python3.8 -D MOE_PYTHON_LIBRARY=/opt/conda/envs/qaliboo/lib/libpython3.8.so.1.0"

RUN python setup.py build_ext

CMD ["python", "asyn_test.py", "--help"]

