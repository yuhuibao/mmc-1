FROM ubuntu:16.04

# Initialize the image
# Modify to pre-install dev tools and ROCm packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
  curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list' && \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  libelf1 \
  libnuma-dev \
  build-essential \
  git \
  vim-nox \
  cmake-curses-gui \
  rocm-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# Setup some convenient tools/functionalities
WORKDIR /root
RUN git clone https://github.com/agostini01/dotfiles.git && \
    \
    ln -sf dotfiles/.gitignore_global .gitignore_global && \
    \
    curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim && \
    \
    ln -sf dotfiles/.vimrc            .vimrc && \
    ln -sf dotfiles/.ctags            .ctags && \
    ln -sf dotfiles/.inputrc          .inputrc && \
    \
    git clone https://github.com/tmux-plugins/tpm /root/.tmux/plugins/tpm && \
    ln -sf dotfiles/.tmux.conf        .tmux.conf

RUN echo "PS1='\[\033[01;31m\][\[\033[01;30m\]\u@\h\[\033[01;36m\] \W\[\033[01;31m\]]\$\[\033[00m\] '" >> .bashrc

# Clonning the interesting projects
RUN git clone https://github.com/yuhuibao/mmc-1.git
WORKDIR ./mmc-1    
RUN git checkout mmcl 
RUN apt update && apt-get install opencl-headers
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/opencl/lib/x86_64
RUN echo "echo 'Hello Yuhui'" >> ~/.bashrc && \
    echo "echo ' '" >> ~/.bashrc && \
    echo "echo 'You are awesome'" >> ~/.bashrc && \
    echo "echo ' '" >> ~/.bashrc 
    
# Change to the correct directory
WORKDIR  ./src
RUN make
ENV PATH "${PATH}:/root/mmc-1/src/bin"
WORKDIR ../examples/validation
CMD mmcl -f cube2.inp -s cube2 -n 1e8 -b 0 -D TP -M G -F bin
