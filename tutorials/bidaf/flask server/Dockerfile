FROM nvcr.io/nvidia/pytorch:18.12.1-py3

RUN apt-get update && \
    apt-get install -y xvfb && \
    apt-get install -y python-opengl

RUN pip install numba && \
    pip install allennlp && \
    pip install fastai && \
    pip install flask && \
    pip install beautifulsoup4 && \
    pip install lxml && \
    pip install google

CMD echo "Flask Environment Created"
