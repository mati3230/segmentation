# Segmentation Environment

Custom gym environment for the segmentation of point clouds with reinforcement learning. A point cloud is segmented by the region growing algorithm. An action consist of 6 parameters. The parameters are: 

* Seed Point X: X coordinate of the seed point in the point cloud.
* Seed Point Y: Y coordinate of the seed point in the point cloud.
* Seed Point Z: Z coordinate of the seed point in the point cloud.
* K: Number of neighbours to grow a region.
* Angle Threshold: Threshold for the angle resulting from the dot product of the normal of the seed point and a region candidate point. Candidate point with a lower angle than this threshold will be added to the region.
* Curvature Threshold: If the curvature of a point in the region is smaller than this threshold, it will be considered as seed point. 

Currently, the state of the environment is a point cloud or a voxel grid which can be specified in the Parameters section with the option *point_mode*. 

## Requirements

A python 3.6.8 interpreter is required. All requirements will be installed in the installation section. 

* [Pyntcloud](https://github.com/mati3230/pyntcloud)
* gym>=0.12.0
* open3d>=0.8.0.0
* numpy>=1.14.5
* matplotlib>=3.1.0"

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

## Parameters

| Parameter | Description | Type | Default |
| --- | --- | --- | --- |
| objs_dir | Location of the 'PointcloudScenes' directory where the point clouds are stored as csv file. The columns of the point cloud should be as follows: [x y z nx ny nz s], where *s* is a segment number. Moreover, the first 6 segments should be wall objects. An example of some point clouds can be found in the PointcloudScenes of the [smartsegmentation](https://github.com/mati3230/smartsegmentation) project. | str | "../point_cloud_env/objects" |
| sample_size | How many points should be sampled from the point cloud as observation for the agent. | int | 1024 |
| wall_sample_size | How many points of the sampled point cloud should be points of the wall in percent. | float | 0.3 |
| unlabelled_punishment | Factor to gain or dampen the punishment of unlabelled points. | float | 1.0 |
| false_labels_punishment | Factor to gain or dampen the punishment of errornous points. |float | 1.0 |
| diff_punishment | Segment difference factor that will be applied in the reward calculation. The agent will be punished for estimating to many or less objects. |float | 0.025 |
| max_steps_per_scene | How many segmentation steps can be done to segment the point cloud. Should be greater or equal the number of the objects in your data. | int | 15 |
| debug | Flag to print useful information. | bool | False |
| max_scenes | How many scenes will be considered for the segmentation. | int | 4 |
| scene_mode | The order in which the scenes are presented. Possible values are 'Random' and 'Linear'. | str | "Random" |
| training | Are you train an agent or do you evaluate training results? If training is 'False' the segmentation result will be plotted after an episode is finished. | bool | True |
| point_mode | Determines the representation of the point cloud. If the value 'None' is selected, the point cloud will be represented as sampled point cloud. The query points will be included in the state representation in the 'Query' mode. The mode 'Voxel' will represent the point cloud as voxel grid. | str | "None" |
| voxel_size | The size of the voxel grid (e.g. a size of 60 produces a voxel grid of dimension 60x60x60). | int | 60 |
| voxel_mode | Determines the features of a voxel. If 'None' is selected, a voxel has only a occupied feature. The features occupied, mean normal and mean curvature are available in the 'Custom' mode. | str | "None" |
| single_scenes | If 'True', the segmentation of one scene counts as an episode. If 'False', the segmentation of all scenes account as one episode. | bool | False |
| early_diff | If 'True', the segment difference will be applied after more segmentation steps are applied than objects in the scene. If 'False', the segment difference will be applied at the end of hte episode. | bool | False |
| wall_weight | The weight of the wall objects in the reward calculation. | float | 0.5 |