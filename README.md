# Segmentation Environment

## Requirements

A python 3.6.8 interpreter is required.

* gym>=0.12.0
* open3d>=0.8.0.0
* numpy>=1.14.5
* matplotlib>=3.1.0"

Open3D>=0.8.0.0 has to be currently compiled from source. Therefore, we provide a compiled version for 
[Ubuntu 18.04.3](https://nextcloud.mirevi.medien.hs-duesseldorf.de/index.php/s/bw8kATaLcieNE9q).

## Installation

Tested on Ubuntu 18.04.3. 

### Ubuntu

1. git clone https://github.com/mati3230/segmentation.git
2. cd segmentation
3. sh ./setup.sh

### Windows

The code is tested on Windows 10. 

1. Clone https://github.com/mati3230/segmentation.git
2. Go to the new directory *segmentation*
3. Clone https://github.com/mati3230/pyntcloud.git
4. Download the Open3D library from https://nextcloud.mirevi.medien.hs-duesseldorf.de/index.php/s/YYRNGZddRj9qaRX and store it in the *segmentation* directory
5. Unzip the files to a new folder in the *segmentation* directory called *o3d_0800_win_10*
  
  Your folder structure should be as follows:
  
  * smartsegmentation
    * segmentation
		* pyntcloud
		* o3d_0800_win_10
    * stable-baselines
		
6. Open a terminal (cmd, Anaconda Prompt) with your Python interpreter, navigate to the *segmentation* directory and install the contents in the *o3d_0800_win_10* folder with: *pip install ./o3d_0800_win_10*
7. Type *pip install -e ./pyntcloud* In your python terminal