# Hierarchical Convolutional Energy Model (HCE) of V4


As of 2024, the HCE model is written in Keras with a Theano backend. Additionally, custom backends were written in the [Keras for Science](https://github.com/the-moliver/kfs) and a fork of the [keras-contrib](https://github.com/the-moliver/keras-contrib) repositories by Dr. Michael Oliver. Originally, the code was written in Python 2. To preserve this code as it is written, a Nvidia Docker container was created such that the code could be run on modern GPU systems. 

Getting Started (Keras, Python2 version):
-------------------
- Install the [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).
- Modify the `Dockerfile` to point to your setup (necessary files, mounts, etc).
- Build the Docker container. See [Docker build documentation](https://docs.docker.com/reference/cli/docker/image/build/).
- Enter the Docker container.
  - Make sure that you have access to GPUs via the command `nvidia-smi`. This command should output information about the GPUs available on the machine.
  - Activate the `dcetester` conda environment.
  - Test that the model can be successfully constructed using the `test.py` script.
