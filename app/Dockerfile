FROM alpine

COPY requirements.txt /src/

RUN apk --update add --virtual scipy-runtime python3 \
    && apk add --virtual scipy-build \
        build-base python3-dev openblas-dev freetype-dev pkgconfig gfortran \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && pip3 install --no-cache-dir numpy \
    && pip3 install --no-cache-dir matplotlib \
    && pip3 install --no-cache-dir scipy \
    && pip3 install --no-cache-dir -r /src/requirements.txt \
    && apk del scipy-build \
    && apk add --virtual scipy-runtime \
        freetype libgfortran libgcc libpng  libstdc++ musl openblas tcl tk \
&& rm -rf /var/cache/apk/*

WORKDIR /src
