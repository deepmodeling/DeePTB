FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install wget -y && apt-get clean all
RUN apt update && apt install -y --no-install-recommends git

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    /opt/miniconda/bin/conda init bash && \
    rm -rf Miniconda3-latest-Linux-x86_64.sh && \ 
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/miniconda/bin:$PATH

RUN git clone https://github.com/deepmodeling/DeePTB.git
RUN conda create -n deeptb python=3.9 -c conda-forge -y 
RUN conda init 
RUN source activate deeptb 
RUN pip install torch==2.2.1  
RUN cd ./DeePTB && \
    pip install . && \
    cd ..  && \ 
    rm ./DeePTB -r && \
    conda clean --all -y && \
    rm -rf /root/.cache/pip && \
    echo "source activate deeptb" >> ~/.bashrc
