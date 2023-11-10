FROM ubuntu:20.04

ARG VCS_REF
ARG BUILD_DATE

ARG DEBIAN_FRONTEND=noninteractive

# Metadata
LABEL org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.name="c-dev" \
      org.label-schema.build-date=$BUILD_DATE

# build tools, common tools, dev tools, network tools
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gdb cmake && \
    apt-get install -y --no-install-recommends bash procps && \
    apt-get install -y --no-install-recommends vim.tiny git libglib2.0-0 && \
    apt-get install -y --no-install-recommends wget iputils-ping less curl && \
    apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# install python
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip

# install jupyterlab for data analysis and plotting
RUN pip3 install --upgrade pip
RUN pip3 install jupyterlab==3.5.3
RUN pip3 install pandas==2.0.3
RUN pip3 install matplotlib==3.7.3

# password 'admin' for login
ADD jupyter_server_config.json /root/.jupyter/jupyter_server_config.json
    
# add alias
ADD bash_aliases /root/.bash_aliases

# set vim encoding for chinese
RUN echo "set encoding=utf-8" >> /root/.vimrc

# set work dir
WORKDIR /root

# Setup docker entry, docker-entry will call supervisord
ADD docker-entrypoint.sh /bin/docker-entrypoint.sh
RUN chmod a+x /bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

# Expose the Flask port
EXPOSE 8888
