# Hierarchical Convolutional Energy Model (HCE) of V4


As of 2024, the HCE model is written in Keras with a Theano backend. Additionally, custom backends were written in the [Keras for Science](https://github.com/the-moliver/kfs) and a fork of the [keras-contrib](https://github.com/the-moliver/keras-contrib) repositories by Dr. Michael Oliver. Originally, the code was written in Python 2. To preserve this code as it is written, a Nvidia Docker container was created such that the code could be run on modern GPU systems. 

Getting Started (Keras, Python2 version):
-------------------
- Clone this repo.
- Install the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).
- Download the nvidia/cuda base image `cuda9.2-cudnn7-devel-ubuntu16.04.tar`.
- Unpack the nvidia/cuda base image so that you can use it as a base from your `Dockerfile`.
  ```
  docker load < cuda9.2-cudnn7-devel-ubuntu16.04.tar
  ```
- Modify the `Dockerfile` to point to your setup (necessary files, mounts, etc).
- Build the Docker container. See [Docker build documentation](https://docs.docker.com/reference/cli/docker/image/build/).
  ```
  docker build --pull=false --rm -f "Dockerfile" -t hcev4:latest "."
  ```
- Enter the Docker container.
  ```
  docker run -it --gpus all --user hce_user hcev4:latest bash
  ```
  - Make sure that you have access to GPUs via the command `nvidia-smi`. This command should output information about the GPUs available on the machine.
  - Run the model using the `hce_gpu` conda environment.
  - Test that the model can be successfully constructed using the `code/test_model_functional.py` script via `python test_model_functional.py`.
