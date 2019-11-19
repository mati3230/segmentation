#!/bin/bash

sudo apt-get install wget unzip
wget --content-disposition https://nextcloud.mirevi.medien.hs-duesseldorf.de/index.php/s/bw8kATaLcieNE9q/download
unzip o3d_0800_ubuntu_18043.zip -d o3d_0800_ubuntu_18043
cd o3d_0800_ubuntu_18043
pip install .
cd ..
pip install -e .
