pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
pip install nvidia-nvshmem-cu13
export NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
TORCH_CUDA_ARCH_LIST="10.0" pip install . --no-build-isolation 2>&1 | tee log

