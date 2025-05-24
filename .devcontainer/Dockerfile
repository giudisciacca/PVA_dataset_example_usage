FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-tk tk && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

#RUN pip install torch==2.2.2+cu122 torchvision==0.17.2+cu122 --extra-index-url https://download.pytorch.org/whl/cu122


CMD ["/bin/bash"]