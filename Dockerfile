FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    bzip2 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1 \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    /bin/bash ~/miniforge.sh -b -p $CONDA_DIR && \
    rm ~/miniforge.sh

# Add conda to PATH
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Create MAtCha directory
WORKDIR /workspace/MAtCha
RUN chmod -R 777 /workspace/MAtCha
RUN git clone https://github.com/Anttwo/MAtCha.git .

# Set environment variables RTX A6000, L40, A100
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CPATH=/usr/local/cuda/include
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9+PTX"

# Activate environment and install dependencies
RUN python install.py --env_name matcha
RUN python download_checkpoints.py

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "matcha", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]
