FROM almalinux:8

# Install tools
RUN dnf clean all && \
    dnf -y update && \
    dnf -y install \
        sudo \
        curl \
        wget \
        git \
        vim \
        nano \
        iputils \
        pciutils \
        which \
        lsof \
        htop && \
    dnf clean all

CMD ["/bin/bash"]