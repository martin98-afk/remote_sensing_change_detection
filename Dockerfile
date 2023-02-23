FROM ubuntu:latest
MAINTAINER jiangmoo

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

RUN sed -i "s/security.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    apt-get clean && apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev \
    python3-dev libevent-dev libhiredis-dev libpq-dev libjpeg-dev libmysqlclient-dev \
    libsasl2-dev libldap2-dev vim nmap python3-pip tzdata

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# RUN wget -O /tmp/Python-3.10.5.tar.xz https://www.python.org/ftp/python/3.10.5/Python-3.10.5.tar.xz && \
#     tar -xf /tmp/Python-3.10.5.tar.xz -C /tmp

RUN mkdir -p /scmdb/backend

# WORKDIR /tmp/Python-3.10.5

# RUN ./configure --prefix=/usr/local/lib/python3.10.5/ && \
#     make -j 4 && make -j 4 install

# RUN ln -s /usr/local/lib/python3.10.5/bin/python3.10 /usr/bin/python3.10 && \
#     rm -f /usr/bin/python3 && \
#     ln -s /usr/local/lib/python3.10.5/bin/python3.10 /usr/bin/python3

WORKDIR /scmdb/backend

# RUN rm -rf /tmp/Python-3.10.5.tar.xz && rm -rf /tmp/Python-3.10.5

COPY ./requirements.txt /scmdb/backend

RUN pip3 install --index-url https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir -r requirements.txt

# RUN ln -s /usr/local/lib/python3.10.5/bin/uwsgi /usr/bin/uwsgi

CMD ["bash"]