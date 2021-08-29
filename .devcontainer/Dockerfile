FROM ubuntu

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV HOME=/home/$USERNAME

RUN apt-get install && apt-get update
RUN apt-get install -y \
    build-essential \
    curl \
    git \
    wget

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN wget -O /tmp/Mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
RUN bash /tmp/Mambaforge.sh -bp $HOME/.conda

ENV PATH=$HOME/.conda/bin:$PATH

RUN conda init bash
RUN pip install \
    bandit \
    flake8 \
    mypy

RUN pip install \
    numpy \
    pytest \
    jupyter \
    matplotlib \
    pip \
    pytest \
    pyyaml \
    torchvision \
    torch
