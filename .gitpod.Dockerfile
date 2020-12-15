FROM gitpod/workspace-full

USER root

# Install util tools.
RUN apt-get update \
 && apt-get install -y \
  apt-utils \
  aria2 \
  git \
  less \
  lcov \
  neofetch \
  neovim \
  asciinema \
  tmux \
  w3m \
  wget

RUN pip3 install --upgrade pip \
    && pip3 install \
	decorator \
	networkx \
	numpy \
	numexpr \
	scipy \
	pre-commit \
	mypy \
	codecov \
	coverage \
	hypothesis \
	pytest \
	pytest-cov \
	pytest-benchmark \
	jupyter \
	jupyterlab \
	matplotlib

USER gitpod

# Install custom tools, runtime, etc. using apt-get
# For example, the command below would install "bastet" - a command line tetris clone:
#
# RUN sudo apt-get -q update && #     sudo apt-get install -yq bastet && #     sudo rm -rf /var/lib/apt/lists/*
#
# More information: https://www.gitpod.io/docs/42_config_docker/
