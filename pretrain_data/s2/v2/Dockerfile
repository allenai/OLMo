FROM ghcr.io/allenai/cuda:11.5-cudnn8-dev-ubuntu20.04-v1.2.2

WORKDIR /app

# update image
RUN apt-get update && apt-get clean;

# copy requirements file
COPY . .

# Install python dependencies
RUN pip install cython && \
    pip install pip --upgrade && \
    pip install pyopenssl --upgrade && \
    pip install torch==1.13.1 torchvision==0.14.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116 && \
    pip install -r requirements.txt

ENTRYPOINT [ "/bin/bash", "-l" ]
