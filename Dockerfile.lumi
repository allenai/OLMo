FROM rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN apt-get clean
