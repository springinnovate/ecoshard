FROM conda/miniconda3

# this builds natcap/ecoshard-docker-test, should be run in the same directory
# as the ecoshard-docker repository root like so:
# docker run -it --rm -v `cwd`:/usr/local/ecoshard natcap/ecoshard-docker-test:[TAG]
# the [TAG] is defined in the .travis.yml file and the latest in VERSION

USER root
RUN apt-get update \
&& apt-get install -y \
    build-essential \
    git \
    libspatialindex-c4v5 \
    mercurial \
&& rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN conda create -y --name py37 python=3.7 && conda clean -a -y
RUN conda run -v -n py37 conda install -c conda-forge gdal=2.4.1
RUN conda run -v -n py37 pip install --no-cache-dir \
    pytest \
    pytest-cov \
    numpy \
    taskgraph && conda clean -a -y

RUN conda init bash && echo "source activate py37" > ~/.bashrc
WORKDIR /usr/local/ecoshard
RUN echo "python setup.py install && \
    pytest -xv tests" > /usr/local/run_ecoshard_tests.sh
ENTRYPOINT /bin/bash -xi /usr/local/run_ecoshard_tests.sh
