# Hough-Transform
Hough Transform implementation in Python

## Table of contents
* [General information](#general-information)
* [Requirements](#requirements)
* [Installation](#installation)
* [Data](#data)
* [Citation](#citation)

## General information

The repository contains Hough Transform in 3D and 2D for point clouds. The code allows to perform point cloud rotation alignment. 

## Requirements

* Python 
  
Libraries:
* matplotlib
* numba
* numpy
* pandas

## Installation

Install necessary packages in the environment:
```yaml
pip install -r requirements.txt
```
To perform Hough Transform on examples run:
```yaml
python hough_transform.py
```

## Data

The repository contains both synthetic and real data from Velodyne VLP-16 LiDAR sensor. 
Synthetic point clouds:
* point
* square
* rectangle
* triangle
* plane
* cube
* prism
* pyramid

The file generating synthetic data is provided and can be modified according to needs. To generate the data run:
```yaml
python generate_synthetic_point_cloud.py
```

Real data examples are the corridor files. 

## Citation

Thank you for citing [our Hough Transform paper](https://doi.org/10.1109/MMAR62187.2024.10680831) if you use any of this code:

```yaml
@INPROCEEDINGS{10680831,
  author={Koszyk, Joanna and Ambroziński, Łukasz and Łabędź, Piotr},
  booktitle={2024 28th International Conference on Methods and Models in Automation and Robotics (MMAR)}, 
  title={Hough Transform for Detection of 3D Point Cloud Rotation}, 
  year={2024},
  volume={},
  number={},
  pages={223-228},
  keywords={Point cloud compression;Performance evaluation;Accuracy;Three-dimensional displays;Simultaneous localization and mapping;Shape;Noise;Hough transform;point cloud;LiDAR;localization;map alignment;robotics},
  doi={10.1109/MMAR62187.2024.10680831}}
```
