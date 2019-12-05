#!/bin/bash

sudo apt-get install wget unzip

VERSION=18043
SUPPORTED=1

V="$(lsb_release -rs)"
FILE="o3d_0800_ubuntu_${VERSION}.zip"
LINK="https://nextcloud.mirevi.medien.hs-duesseldorf.de/index.php/s/bw8kATaLcieNE9q/download" # 18.04.3

if [ ${V} = "19.10" ]
then
    VERSION=1910
    LINK="https://nextcloud.mirevi.medien.hs-duesseldorf.de/index.php/s/zqeb4JWfma9icdd/download"
else
    SUPPORTED=0
fi
FILE="o3d_0800_ubuntu_${VERSION}.zip"

if [ ! -f ${FILE} ]
then
    wget --content-disposition ${LINK}
fi

if [ ${SUPPORTED} = 1 ]
then
    DIR="o3d_0800_ubuntu_${VERSION}"
    if [ ! -d ${DIR} ]
    then
        unzip o3d_0800_ubuntu_${VERSION}.zip -d ${DIR}
    fi
    cd o3d_0800_ubuntu_${VERSION}
    pip install .
    cd ..
else
    echo "Your ubuntu version is not supported (supported versions: 1910, 1804). You have to compile and install Open3D by yourself."
fi

DIR="pyntcloud"
if [ ! -d ${DIR} ]
then
    git clone https://github.com/mati3230/pyntcloud.git
fi
cd pyntcloud
pip install -e .
cd ..

pip install -e .
