FROM python:3.10
RUN pip install torch
RUN pip install torchsort 
RUN pip install numpy scipy spglib matplotlib 
RUN pip install ase   
RUN pip install pytest
RUN pip install future
RUN pip install pyyaml
