# Example of Dockerfile with cuda 10.2, pytorch, jupyterlab...
FROM nvidia/cuda:10.2-base
CMD nvidia-smi
FROM pytorch/pytorch

RUN conda update conda
RUN apt-get update && apt-get install apt-file -y && apt-file update && \
 apt-get install -y git build-essential curl wget software-properties-common \
 zip unzip libssl-dev tmux nano graphviz graphviz-dev
RUN python -m pip install --upgrade pip &&  pip install tensorboard pandas scikit-learn gym moz_sql_parser psycopg2_binary optuna pygraphviz networkx
WORKDIR /home/
# COPY reqs.txt /home/reqs.txt
# RUN pip install -r ./reqs.txt

CMD ["/bin/bash"]