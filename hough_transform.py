"""
Implementation of Hough Transform in 2D and 3D for point clouds has been presented.

It was prepared to perform Hough Transform on point clouds passed as csv files to the function.

Algorithm can be perform either on set of point clouds contained in a folder or a single point cloud.

To perform on a folder run:
run_hough_transform_folder("input_folder")
To perform on a single file run:
run_hough_transform_single_cloud("input_file.csv")
There are optional parameters that can be set accordingly to needs of the user.

"""

__author__ = "Joanna Koszyk"
__contact__ = "jkoszyk@agh.edu.pl"
__copyright__ = "Copyright 2023, AGH"
__date__ = "2023/06/23"
__email__ = "jkoszyk@agh.edu.pl"
__version__ = "1.0.0"

import fnmatch
import os
import numpy as np
from numba import jit
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
# import time

matplotlib.rc('font', size=12)
matplotlib.rc('axes', titlesize=16)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


@jit(nopython=True)
def calculate_hough_space(pcld):
    # (x, y, z) -> (theta, phi, rho)
    # returns point cloud in hough space
    # function is not used in current version of the program
    pcld2 = np.asarray(pcld)
    for i, p in enumerate(pcld):

        x, y, z = p
        # d_2d = math.sqrt((x ** 2 + y ** 2))
        rho = math.sqrt((x ** 2 + y ** 2 + z ** 2))
        theta = math.atan((y/x))
        phi = math.asin((z/rho))

        pcld2[i] = [theta, phi, rho]

    return pcld2


@jit(nopython=True)
def calculate_rho_extremum(pcld0):
    # calculate maximum and minimum value of rho for 3D point cloud
    # to some extent extremum rho can be an indicator how to define rho_start and rho_stop parameters
    rho = np.sqrt(np.add(np.square(pcld0[:, 0]), np.square(pcld0[:, 1]), np.square(pcld0[:, 2])))
    # find maximum and minimum rho
    max_rho1 = np.max(rho)
    min_rho1 = np.min(rho)
    return min_rho1, max_rho1


@jit(nopython=True)
def calculate_rho_extremum_2d(pcld0):
    # calculate maximum and minimum value of rho for 2D point cloud
    # to some extent extremum rho can be an indicator how to define rho_start and rho_stop parameters
    rho = np.sqrt(np.add(np.square(pcld0[:, 0]), np.square(pcld0[:, 1])))
    # find maximum and minimum rho
    max_rho1 = np.max(rho)
    min_rho1 = np.min(rho)
    return min_rho1, max_rho1


@jit(nopython=True)
def calculate_xyz_values(theta, fi, rho):
    z1 = rho * math.sin(fi)
    y1 = rho * math.cos(fi) * math.sin(theta)
    x1 = rho * math.cos(fi) * math.cos(theta)
    return x1, y1, z1


@jit(nopython=True)
def calculate_xyz_numpy_values(theta_array1, fi_array1, rho_array1):
    z1 = np.multiply(rho_array1, np.sin(fi_array1))
    y1 = np.multiply(rho_array1, np.cos(fi_array1), np.sin(theta_array1))
    x1 = np.multiply(rho_array1, np.cos(fi_array1), np.cos(theta_array1))
    point_cloud1 = np.hstack(x1, y1, z1)
    return point_cloud1


@jit(nopython=True)
def hough_trans3d(pcld_xyz, max_rho1, min_rho1, theta_step1=0.01 * math.pi, fi_step1=0.02 * math.pi,
                  rho_step1=0.05, threshold=0.04):

    # Hough Transform Implementation

    # version 1 - theta_array, fi_array and rho_array are created inside the function
    # could be useful if theta_step or fi_step (or grid) changed while running
    # in most cases (but not always) - the slowest method

    # goto hough_trans3d_3()

    """
    Algorithm:

    1: for all points in pi in point set P do
    2: for all cells (θ, φ, ρ) in accumulator A do
    3: if point pi lies on the plane defined by (θ, φ, ρ)
    then
    4: increment cell A(θ, φ, ρ)
    5 : end if
    6 : end for
    7 : end for
    8 : search for the most prominent cells in the
    accumulator, that define the detected planes in P

    The 3D Hough Transform for Plane Detection in Point Clouds: A Review and a new Accumulator Design
    Dorit Borrmann • Jan Elseberg • Kai Lingemann • Andreas Nüchter
    """

    # px ⋅ cosθ ⋅ sinϕ + py ⋅ sinϕ ⋅ sinθ + pz ⋅ cosϕ = ρ

    theta_array1 = np.arange(-math.pi, math.pi, theta_step1)
    fi_array1 = np.arange(-math.pi, math.pi, fi_step1)
    rho_array1 = np.arange(min_rho1, max_rho1, rho_step1)
    len_theta = len(theta_array1)
    len_fi = len(fi_array1)
    len_rho = len(rho_array1)

    acc = np.zeros((len_theta, len_fi, len_rho), dtype="int64")

    for p in pcld_xyz:
        x, y, z = p
        # for theta1, fi1, rho1 in zip(theta_array, fi_array, rho_array):
        for j, theta1 in enumerate(theta_array1):
            for k, fi1 in enumerate(fi_array1):
                rho2 = x * math.cos(theta1) * math.sin(fi1) + y * math.sin(fi1) * math.sin(theta1) + z * math.cos(fi1)
                for l1, rho1 in enumerate(rho_array1):

                    # print(theta1,",", fi1, ",", rho1, ",", rho2)
                    if abs(rho1 - rho2) < threshold:
                        # x1, y1, z1 = calculate_xyz_values(theta1, fi1, rho1)

                        acc[j, k, l1] += 1

    ind1 = np.argmax(acc)

    return acc, ind1


@jit(nopython=True)
def hough_trans3d_2(pcld_xyz, max_rho1, min_rho1, rho_step1, theta_array1, fi_array1, threshold=0.04):

    # Hough Transform Implementation

    # version 2 - rho_array is created inside the function

    # goto hough_trans3d_3()

    """
    Algorithm:

    1: for all points in pi in point set P do
    2: for all cells (θ, φ, ρ) in accumulator A do
    3: if point pi lies on the plane defined by (θ, φ, ρ)
    then
    4: increment cell A(θ, φ, ρ)
    5 : end if
    6 : end for
    7 : end for
    8 : search for the most prominent cells in the
    accumulator, that define the detected planes in P

    The 3D Hough Transform for Plane Detection in Point Clouds: A Review and a new Accumulator Design
    Dorit Borrmann • Jan Elseberg • Kai Lingemann • Andreas Nüchter
    """

    # px ⋅ cosθ ⋅ sinϕ + py ⋅ sinϕ ⋅ sinθ + pz ⋅ cosϕ = ρ
    rho_array1 = np.arange(min_rho1, max_rho1, rho_step1)
    len_theta = len(theta_array1)
    len_fi = len(fi_array1)
    len_rho = len(rho_array1)

    acc = np.zeros((len_theta, len_fi, len_rho), dtype="int64")

    for p in pcld_xyz:
        x, y, z = p
        # for theta1, fi1, rho1 in zip(theta_array, fi_array, rho_array):
        for j, theta1 in enumerate(theta_array1):
            for k, fi1 in enumerate(fi_array1):
                rho2 = x * math.cos(theta1) * math.sin(fi1) + y * math.sin(fi1) * math.sin(theta1) + z * math.cos(fi1)
                for l1, rho1 in enumerate(rho_array1):

                    # print(theta1,",", fi1, ",", rho1, ",", rho2)
                    if abs(rho1 - rho2) < threshold:
                        # x1, y1, z1 = calculate_xyz_values(theta1, fi1, rho1)

                        acc[j, k, l1] += 1

    ind1 = np.argmax(acc)

    return acc, ind1


@jit(nopython=True)
def hough_trans3d_3(pcld_xyz, theta_array1, fi_array1, rho_array1, threshold=0.04):

    # Hough Transform Implementation
    # This version is used in the code
    # version 3 - theta_array, fi_array and rho_array are passed to the function

    # statistically the best, but small sample was taken into account

    """
    Hough Transform in 3D for point clouds

    Algorithm:

    1: for all points in pi in point set P do
    2: for all cells (θ, φ, ρ) in accumulator A do
    3: if point pi lies on the plane defined by (θ, φ, ρ)
    then
    4: increment cell A(θ, φ, ρ)
    5 : end if
    6 : end for
    7 : end for
    8 : search for the most prominent cells in the
    accumulator, that define the detected planes in P

    The 3D Hough Transform for Plane Detection in Point Clouds: A Review and a new Accumulator Design
    Dorit Borrmann • Jan Elseberg • Kai Lingemann • Andreas Nüchter
    """
    # px ⋅ cosθ ⋅ sinϕ + py ⋅ sinϕ ⋅ sinθ + pz ⋅ cosϕ = ρ
    len_theta = len(theta_array1)
    len_fi = len(fi_array1)
    len_rho = len(rho_array1)

    acc = np.zeros((len_theta, len_fi, len_rho), dtype="int64")

    for p in pcld_xyz:
        x, y, z = p
        # for theta1, fi1, rho1 in zip(theta_array, fi_array, rho_array):
        for j, theta1 in enumerate(theta_array1):
            for k, fi1 in enumerate(fi_array1):
                rho2 = x * math.cos(theta1) * math.sin(fi1) + y * math.sin(fi1) * math.sin(theta1) + z * math.cos(fi1)
                for l1, rho1 in enumerate(rho_array1):

                    # print(theta1,",", fi1, ",", rho1, ",", rho2)
                    if abs(rho1 - rho2) < threshold:
                        # x1, y1, z1 = calculate_xyz_values(theta1, fi1, rho1)

                        acc[j, k, l1] += 1

    ind1 = np.argmax(acc)

    return acc, ind1


@jit(nopython=True)
def hough_transform_in_2d(pcld_xy, theta_array1, rho_array1, threshold):

    """
    Hough Transform in 2D for point clouds
    Solution is corresponding to the one for 3D.

    Equation for 2D:
    ρ = x cos(theta) + y sin(theta)
    """

    # theta_array1 = np.arange(0, math.pi, theta_step1)
    # rho_array1 = np.arange(min_rho1, max_rho1, rho_step1)

    len_theta = len(theta_array1)
    len_rho = len(rho_array1)
    acc = np.zeros((len_rho, len_theta), dtype="int64")

    for p in pcld_xy:
        x, y = p
        for j, theta1 in enumerate(theta_array1):
            rho2 = x * math.cos(theta1) + y * math.sin(theta1)
            for k, rho1 in enumerate(rho_array1):
                if abs(rho1 - rho2) < threshold:
                    acc[k, j] += 1
    ind1 = np.argmax(acc)
    return acc, ind1


@jit(nopython=True)
def get_hough_plane_values_3d(ind3, theta_array1, fi_array1, rho_array1):

    # get theta, fi and rho values for accumulator max value
    theta_max0 = theta_array1[ind3[0]]
    fi_max0 = fi_array1[ind3[1]]
    rho_max0 = rho_array1[ind3[2]]

    return theta_max0, fi_max0, rho_max0


@jit(nopython=True)
def get_hough_plane_values_2d(ind3, theta_array1, rho_array1):

    # get theta and rho values for accumulator max value
    theta_max0 = theta_array1[ind3[0]]
    rho_max0 = rho_array1[ind3[1]]

    return theta_max0, rho_max0


# @jit(nopython=True)
def rotate_point_cloud(rotation_angle1, x1, y1, z1):

    # rotation of a point cloud around Z axis with theta_angle from Hough Transform
    # version 1

    rotation_matrix1 = np.array([[math.cos(rotation_angle1), -math.sin(rotation_angle1), 0],
                                [math.sin(rotation_angle1), math.cos(rotation_angle1), 0],
                                [0, 0, 1]])

    xyz_matrix = np.vstack((x1, y1, z1))
    matrix_shape = np.shape(xyz_matrix)

    new_xyz_matrix = np.empty(matrix_shape)

    for i in range(matrix_shape[1]):
        xyz = xyz_matrix[:, i]

        xyz = xyz.transpose()
        new_xyz = rotation_matrix1 @ xyz
        new_xyz_matrix[:, i] = new_xyz
        # if i == 1:
        #     print("xyz:", xyz)
        #     print("rot:", rotation_matrix)
        #     print("new_xyz: ", new_xyz)

    rotated_pcld0 = new_xyz_matrix.transpose()

    return rotated_pcld0


# @jit(nopython=True)
def rotate_point_cloud2(rotation_angle1, pcld):
    # rotation of a point cloud around Z axis with theta_angle from Hough Transform
    # version 2
    rotation_matrix1 = np.array([[math.cos(rotation_angle1), -math.sin(rotation_angle1), 0],
                                [math.sin(rotation_angle1), math.cos(rotation_angle1), 0],
                                [0, 0, 1]])
    # Use of unsupported NumPy function 'numpy.matmul'
    rotated_pcld0 = np.matmul(rotation_matrix1, pcld.transpose()).transpose()

    return rotated_pcld0


@jit(nopython=True)
def rotate_point_cloud3(rotation_matrix0, pcld0):

    # rotation of a point cloud around Z axis with theta_angle from Hough Transform
    # version 3 - is accelerated with numba.jit

    matrix_shape = pcld0.shape
    new_xyz_matrix = np.empty(matrix_shape)

    for i in range(matrix_shape[0]):
        xyz = pcld0[i, :]
        xyz = np.transpose(xyz)
        new_xyz = np.dot(rotation_matrix0, xyz)
        new_xyz_matrix[i, :] = new_xyz

    return new_xyz_matrix


def translate_point_cloud(pcld,  rho1):
    pass


# @jit(nopython=True)
def cut_point_cloud(pcld, min_z, max_z):

    # cut floor and ceiling
    # The point cloud will be limited on Z axis. If both minimal and maximal Z value were not set - use default values
    # - the point cloud will be cut below 0.01 and above 2.0 meters. If only one value was set the other side of
    # the point cloud will not be affected.

    if min_z is None and max_z is None:
        # if values were not set - use default values
        min_z = 0.01
        max_z = 2.0

    # validate min_z and max_z values
    if min_z is not None and max_z is not None:
        assert min_z < max_z, "minimum value is higher than maximum"

    z0 = pcld[:, 2]
    if min_z is None and max_z is not None:
        # cut ceiling only
        pcld0 = pcld[np.where(z0 < max_z)]
    elif min_z is not None and max_z is None:
        # cut floor only
        pcld0 = pcld[np.where(z0 > min_z)]
    else:
        pcld0 = pcld[np.where(z0 > min_z)]
        z0 = pcld0[:, 2]
        pcld0 = pcld0[np.where(z0 < max_z)]

    return pcld0


def show_accumulator_2d(theta_array1, rho_array1, a, name=None):
    # Visualize the accumulator in 2D
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    x1, y1 = np.meshgrid(theta_array1, rho_array1, indexing='xy')
    cs = ax1.contourf(x1, y1, a, 50, cmap='Blues')
    ax1.set_xlabel('theta (rad)')
    ax1.set_ylabel('rho (m)')
    cbar = plt.colorbar(cs)
    # print(np.max(a))
    if name is not None:
        name = name[:-4]
        print(name)
        plt.savefig(f'{name}.eps', format='eps')
        # # save data x, y, a
        # print(x1.shape)
        # print(y1.shape)
        # print(a.ravel().shape)
        # print(theta_array1.shape)
        # print(rho_array1.shape)
        # arr = np.vstack((x1.ravel(), a.ravel()))
        # print(arr.shape)
        # arr = arr.reshape((-1, 2))
        #
        # df = pd.DataFrame(arr)
        # df.to_csv(f"{name}_accumulator.csv")
    plt.show()


def show_accumulator_3d(theta_array1, fi_array1, rho_array1, a, name=None):
    # Visualize the accumulator in 3D
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    x1, y1, z1 = np.meshgrid(theta_array1, fi_array1, rho_array1)
    # cs = ax1.contourf(x1, y1, z1, a, 50, cmap='viridis')  # cmap = 'plasma'
    ax1.scatter(x1.ravel(), y1.ravel(), z1.ravel(), c=a.ravel(), s=1, cmap='viridis')
    ax1.set_xlabel('theta (rad)')
    ax1.set_ylabel('fi (rad)')
    ax1.set_zlabel('rho (m)')
    # cbar = plt.colorbar(cs)
    if name is not None:
        plt.savefig(f'{name}.eps', format='eps')
        # save data x, y, z, a
    plt.show()


def show_3d_point_cloud(pcld_coordinates1):
    # visualize point cloud in 3D
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(pcld_coordinates1[:, 0], pcld_coordinates1[:, 1], pcld_coordinates1[:, 2])
    ax1.axis("equal")
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    plt.show()


def show_2d_point_cloud(pcld_coordinates1):
    # visualize point cloud in 2D
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.scatter(pcld_coordinates1[:, 0], pcld_coordinates1[:, 1])
    ax1.axis("equal")
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    plt.show()


def run_hough_transform(path1, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                        visualize_accumulator, rho_step, threshold, point_cloud_cut_floor, point_cloud_cut_ceiling,
                        *args, fi_array=None):

    # read point cloud from file
    cloud0 = pd.read_csv(path1, header=None)
    pcld1 = np.array(cloud0)

    # if cloud is cut then it is performed before (possible) transferring to 2D domain due to possibility of
    # removing outliers or incorrect data
    if cut_cloud or (point_cloud_cut_floor is not None) or (point_cloud_cut_ceiling is not None):
        # cut floor and ceiling - default from 0.01 to 2.0 meters
        pcld1 = cut_point_cloud(pcld1, point_cloud_cut_floor, point_cloud_cut_ceiling)

    if for_3d:
        pcld_coordinates_3 = pcld1[:, 0:3]
    else:
        pcld_coordinates_3 = pcld1[:, 0:2]

    pcld_coordinates_2 = pcld_coordinates_3.copy()
    pcld_coordinates = pcld_coordinates_3.copy()

    if visualize_point_cloud:
        if for_3d:
            show_3d_point_cloud(pcld_coordinates)
        else:
            show_2d_point_cloud(pcld_coordinates)

    # calculate max and min rho
    if for_3d:
        min_rho, max_rho = calculate_rho_extremum(pcld_coordinates_2)
    else:
        min_rho, max_rho = calculate_rho_extremum_2d(pcld_coordinates_2)

    # # first hough method
    # # count time of hough transform
    # start = time.time()
    #
    # # perform hough transform on original point cloud
    # A, ind = hough_trans3d(pcld_coordinates_3, max_rho, min_rho, theta_step1=theta_step,
    #                      fi_step1=fi_step, rho_step1=rho_step)
    # end1 = time.time()

    # # second hough method - more efficient
    # # count time of hough transform
    # start4 = time.time()
    # # perform hough transform on original point cloud
    # A, ind = hough_trans3d_2(pcld_coordinates_3, max_rho, min_rho, rho_step, theta_array, fi_array)
    # end5 = time.time()

    # start3 = time.time()
    if args:
        rho_array = args[0]
    else:
        rho_array = np.arange(min_rho/2, 2*max_rho, rho_step)

    if for_3d:
        A, ind = hough_trans3d_3(pcld_coordinates_3, theta_array, fi_array, rho_array, threshold)
    else:
        A, ind = hough_transform_in_2d(pcld_coordinates_3, theta_array, rho_array, threshold)

    # end4 = time.time()

    # transform single index into indices
    # function unravel_index is unknown by numba.jit
    ind2 = np.unravel_index(ind, A.shape)
    # print(ind2)
    if for_3d:
        theta_max, fi_max, rho_max = get_hough_plane_values_3d(ind2, theta_array, fi_array, rho_array)
    else:
        theta_max, rho_max = get_hough_plane_values_2d(ind2, theta_array, rho_array)
    # end2 = time.time()

    # print(f"comparison of performance of 3 methods 1: {abs(start - end1)}, "
    #       f"2: {abs(start4 - end5)}, 3: {abs(start3 - end4)}")

    # print(f"hough transform: {abs(start-end1)}, hough and theta value: {abs(start-end2)}")
    # print(f"theta: {theta_max} - {math.degrees(theta_max)}, fi: {fi_max} - {math.degrees(fi_max)}, rho: {rho_max}")

    if visualize_accumulator:
        # Visualize the accumulator in 3D
        if for_3d:
            show_accumulator_3d(theta_array, fi_array, rho_array, A, name=path1)
        else:
            show_accumulator_2d(theta_array, rho_array, A, name=path1)

    # check if point cloud should be rotated and saved
    if output_dir is not None:

        # save files in another directory
        # check variable output_dir is a directory or name of a file
        # in first case - the hough transform is performed on multiple files
        # else it is a single point_cloud

        if os.path.isdir(output_dir):
            # change name of the file
            path_split = os.path.split(path1)
            csv_file = path_split[-1]
            csv_file2 = csv_file[:-4] + "_rotated.csv"
            path2 = os.path.join(output_dir, csv_file2)
        else:
            path2 = output_dir

        # print(path2)
        # rotated_pcld = rotate_point_cloud(-theta_max, x0, y0, z0)
        # rotated_pcld = rotate_point_cloud2(-theta_max, pcld_coordinates)
        # start2 = time.time()
        rotation_angle = -theta_max
        print(rotation_angle)
        if for_3d:
            rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle), 0],
                                        [math.sin(rotation_angle), math.cos(rotation_angle), 0],
                                        [0, 0, 1]])
        else:
            rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                                        [math.sin(rotation_angle), math.cos(rotation_angle)]])

        rotated_pcld = rotate_point_cloud3(rotation_matrix, pcld_coordinates)
        point_cloud_4 = translate_point_cloud(rotated_pcld, rho_max)
        # end3 = time.time()
        # print(f"rotation time: {abs(start2-end3)}")
        # save point cloud (x, y, z) to csv file
        df2 = pd.DataFrame(point_cloud_4)
        df2.to_csv(path2, header=False, index=False)

        # # save files in another directory
        # dir3_name = "korytarz_pojedyncze_rotated_2"
        # # change name of the file
        # csv_file3 = csv_file[:-4] + "_rotated.csv"
        # path3 = os.path.join(dir3_name, csv_file3)
        # df3 = pd.DataFrame(rotated_pcld_2)
        # df3.to_csv(path3)

        print(path2)


def run_hough_transform_folder(input_dir, output_dir, for_3d, cut_cloud, visualize_point_cloud, visualize_accumulator,
                               theta_step, fi_step, rho_step, threshold, theta_start, theta_stop, fi_start, fi_stop,
                               rho_start, rho_stop, point_cloud_cut_floor, point_cloud_cut_ceiling):

    assert threshold < rho_step, "threshold must be smaller than rho_step"

    # assert os.path.isdir(input_dir), "nonexistent directory"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print("Directory '% s' created" % output_dir)

    # load datasets from directory
    # list all csv files
    dir_list = fnmatch.filter(os.listdir(input_dir), "*.csv")

    theta_array = np.arange(0, 2*math.pi, theta_step)
    if (theta_start is not None and theta_stop is not None) and (theta_start < theta_stop):
        theta_array = np.arange(theta_start, theta_stop, theta_step)

    if for_3d:
        fi_array = np.arange(0, math.pi * 2, fi_step)
        if (fi_start is not None and fi_stop is not None) and (fi_start < fi_stop):
            fi_array = np.arange(fi_start, fi_stop, fi_step)

    # if **kwargs
    # check for theta_start, theta_stop - keys - if existent -> check if it makes sense
    #

    if (rho_start is not None and rho_stop is not None) and (rho_start < rho_stop):
        rho_array = np.arange(rho_start, rho_stop, rho_step)

    for csv_file in dir_list:
        path = os.path.join(input_dir, csv_file)
        # perform hough transform for each file
        if for_3d:
            # pass fi_array to the function

            # if rho_array was predefined for all point cloud and will not be changed due to the
            # maximum possible value of rho in each point cloud
            if 'rho_array' in locals():
                run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                    visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                    point_cloud_cut_ceiling, rho_array, fi_array=fi_array)
            else:
                run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                    visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                    point_cloud_cut_ceiling, fi_array=fi_array)
        else:
            if 'rho_array' in locals():
                run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                    visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                    point_cloud_cut_ceiling, rho_array)
            else:
                run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                    visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                    point_cloud_cut_ceiling)


def run_hough_transform_single_cloud(path, output_dir, for_3d, cut_cloud, visualize_point_cloud, visualize_accumulator,
                                     theta_step, fi_step, rho_step, threshold, theta_start, theta_stop, fi_start,
                                     fi_stop, rho_start, rho_stop, point_cloud_cut_floor, point_cloud_cut_ceiling):

    assert threshold < rho_step, "threshold must be smaller than rho_step"

    assert os.path.isfile(path), "file is nonexistent"

    # if for_3d:
    #     fi_array = np.arange(0, math.pi * 2, fi_step)
    #     # pass fi_array to the function
    #     run_hough_transform(input_path, output_path, theta_array, for_3d, cut_cloud, visualize_point_cloud,
    #                         visualize_accumulator, rho_step, threshold, fi_array)
    # else:
    #     run_hough_transform(input_path, output_path, theta_array, for_3d, cut_cloud, visualize_point_cloud,
    #                         visualize_accumulator, rho_step, threshold)
    theta_array = np.arange(-math.pi, math.pi, theta_step)
    if (theta_start is not None and theta_stop is not None) and (theta_start < theta_stop):
        theta_array = np.arange(theta_start, theta_stop, theta_step)

    if for_3d:
        fi_array = np.arange(-math.pi, math.pi, fi_step)
        if (fi_start is not None and fi_stop is not None) and (fi_start < fi_stop):
            fi_array = np.arange(fi_start, fi_stop, fi_step)

    # if **kwargs
    # check for theta_start, theta_stop - keys - if existent -> check if it makes sense
    #

    if (rho_start is not None and rho_stop is not None) and (rho_start < rho_stop):
        rho_array = np.arange(rho_start, rho_stop, rho_step)
    if for_3d:
        # pass fi_array to the function

        # if rho_array was predefined for all point cloud and will not be changed due to the
        # maximum possible value of rho in each point cloud
        if 'rho_array' in locals():
            run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                point_cloud_cut_ceiling, rho_array, fi_array=fi_array)
        else:
            run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                point_cloud_cut_ceiling, fi_array=fi_array)
    else:
        if 'rho_array' in locals():
            run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                point_cloud_cut_ceiling, rho_array)
        else:
            run_hough_transform(path, output_dir, theta_array, for_3d, cut_cloud, visualize_point_cloud,
                                visualize_accumulator, rho_step, threshold, point_cloud_cut_floor,
                                point_cloud_cut_ceiling)


# 2D examples
def run_hough_transform_point():
    point_dir = os.path.join("examples", "point.csv")
    hough_transform(point_dir, for_3d=False, cut_cloud=False, visualize_point_cloud=True, visualize_accumulator=True,
                    theta_step=0.001*math.pi, rho_step=0.01, threshold=0.005, rho_start=-10.0, rho_stop=10.0,
                    theta_start=-math.pi, theta_stop=math.pi)


def run_hough_transform_square():
    square_dir = os.path.join("examples", "square.csv")
    output_path = os.path.join("examples", "square_ht2d.csv")
    hough_transform(square_dir, output_dir=output_path, for_3d=False, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-10.0, rho_stop=10.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_rectangle():
    rectangle_dir = os.path.join("examples", "rectangle.csv")
    output_path = os.path.join("examples", "rectangle_ht2d.csv")
    hough_transform(rectangle_dir, output_dir=output_path, for_3d=False, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-7.5, rho_stop=7.5, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_triangle():
    triangle_dir = os.path.join("examples", "triangle.csv")
    output_path = os.path.join("examples", "triangle_ht2d.csv")
    hough_transform(triangle_dir, output_dir=output_path, for_3d=False, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-8.0, rho_stop=8.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_plane():
    plane_dir = os.path.join("examples", "plane.csv")
    output_path = os.path.join("examples", "plane_ht2d.csv")
    hough_transform(plane_dir, output_dir=output_path, for_3d=False, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-5.0, rho_stop=5.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_cube_2d():
    cube_dir = os.path.join("examples", "cube.csv")
    output_dir = os.path.join("examples", "cube_ht2d.csv")
    hough_transform(cube_dir, output_dir=output_dir, for_3d=False, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-8.0, rho_stop=8.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_corridor_2d():
    corridor_dir = os.path.join("examples", "corridor.csv")
    output_dir = os.path.join("examples", "corridor_ht2d.csv")
    hough_transform(corridor_dir, output_dir=output_dir, for_3d=False, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-5.0, rho_stop=5.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_cut_corridor_2d():
    corridor_dir = os.path.join("examples", "corridor.csv")
    output_dir = os.path.join("examples", "cut_corridor_ht2d")
    hough_transform(corridor_dir, output_dir=output_dir, for_3d=False, cut_cloud=True, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.001 * math.pi, rho_step=0.01, threshold=0.005,
                    rho_start=-5.0, rho_stop=5.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


# 3D examples
def run_hough_transform_cube():
    cube_dir = os.path.join("examples", "cube.csv")
    output_path = os.path.join("examples", "cube_ht.csv")
    hough_transform(cube_dir, output_dir=output_path, for_3d=True, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.01*math.pi, fi_step=0.1 * math.pi, rho_step=0.05,
                    threshold=0.04, rho_start=-15.0, rho_stop=15.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_prism():
    prism_dir = os.path.join("examples", "prism.csv")
    output_path = os.path.join("examples", "prism_ht.csv")
    hough_transform(prism_dir, output_dir=output_path, for_3d=True, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.05*math.pi, fi_step=0.1 * math.pi, rho_step=0.5,
                    threshold=0.4, rho_start=-15.0, rho_stop=15.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_pyramid():
    pyramid_dir = os.path.join("examples", "pyramid.csv")
    output_path = os.path.join("examples", "pyramid_ht.csv")
    hough_transform(pyramid_dir, output_path, for_3d=True, cut_cloud=False, visualize_point_cloud=True,
                    visualize_accumulator=True, theta_step=0.05*math.pi, fi_step=0.1 * math.pi, rho_step=0.5,
                    threshold=0.4, rho_start=-15.0, rho_stop=15.0, theta_start=-math.pi/2, theta_stop=math.pi/2)


def run_hough_transform_corridor_3d():
    input_dir = os.path.join("examples", "corridor")
    output_dir = os.path.join("examples", "corridor_ht")
    hough_transform(input_dir, output_dir, visualize_point_cloud=True, visualize_accumulator=True)


def run_hough_transform_examples_3d():
    run_hough_transform_cube()
    run_hough_transform_prism()
    run_hough_transform_pyramid()
    run_hough_transform_corridor_3d()


def run_hough_transform_examples_2d():
    run_hough_transform_point()
    run_hough_transform_square()
    run_hough_transform_rectangle()
    run_hough_transform_triangle()
    run_hough_transform_plane()
    run_hough_transform_cube_2d()
    run_hough_transform_corridor_2d()
    run_hough_transform_cut_corridor_2d()


def run_hough_transform_real_data_examples():

    corridor_dir = os.path.join("examples", "corridor")
    output_path1 = os.path.join("examples", "corridor_ht_ex1")
    hough_transform(corridor_dir, output_dir=output_path1, for_3d=False, cut_cloud=True, visualize_point_cloud=False,
                    visualize_accumulator=False, theta_step=0.005*math.pi, fi_step=0.1*math.pi, rho_step=0.7,
                    threshold=0.5, theta_start=None, theta_stop=None, fi_start=None, fi_stop=None, rho_start=None,
                    rho_stop=None, point_cloud_cut_floor=0.3, point_cloud_cut_ceiling=1.3)
    output_path2 = os.path.join("examples", "corridor_ht_ex2")
    hough_transform(corridor_dir, output_dir=output_path2, for_3d=False, cut_cloud=True, visualize_point_cloud=False,
                    visualize_accumulator=False, theta_step=0.01*math.pi, rho_step=0.03, threshold=0.02, theta_start=0,
                    theta_stop=math.pi, fi_start=0, fi_stop=math.pi, rho_start=0, point_cloud_cut_floor=0.1,
                    point_cloud_cut_ceiling=2.0)


def hough_transform(path, output_dir=None, for_3d=True, cut_cloud=False, visualize_point_cloud=False,
                    visualize_accumulator=True, theta_step=0.01 * math.pi, fi_step=0.01 * math.pi, rho_step=0.05,
                    threshold=0.04, theta_start=None, theta_stop=None, fi_start=None, fi_stop=None, rho_start=None,
                    rho_stop=None, point_cloud_cut_floor=None, point_cloud_cut_ceiling=None):

    """
    :param path: path to point clouds or a single one (required).
    :param output_dir: path where point clouds after rotation and translation will be saved (required).

    Optional parameters:
    :param for_3d: mode 2D (False) or 3D (True) (optional, default=True).
    :param cut_cloud: parameter is set if point cloud should be cut before Hough transform. If parameters
                      point_cloud_cut_floor and point_cloud_cut_ceiling are None then point cloud will be cut with
                      default values from 0.01 to 2.0. For 2D Hough Transform setting, point cloud will be cut before
                      flattening a point cloud and performing Hough Transform (optional, default=False).
    :param visualize_point_cloud: plot with original point cloud in chosen projection - (optional, default=False).
    :param visualize_accumulator: plot with accumulator (optional, default=True).
    :param theta_step: theta step in Hough Space (optional, default=0.01*math.pi).
    :param fi_step: fi step in Hough Space. Used only when for_3d=True (optional, default=0.01*math.pi).
    :param rho_step: rho step in Hough Space  (optional, default=0.05).
    :param threshold: value indicating that point P lies on the Hough plane (optional, default=0.04).
    :param theta_start: starting theta value in Hough space. When not defined by the user the starting value is 0
                        (optional, default=None).
    :param theta_stop: stopping theta value in Hough space. When not defined by the user the stopping value is π
                       (optional, default=None).
    :param fi_start: starting fi value in Hough space. When not defined by the user the starting value is 0.
                     Used only when for_3d=True (optional, default=None).
    :param fi_stop: stopping fi value in Hough space. When not defined by the user the stopping value is 2π.
                    Used only when for_3d=True (optional, default=None).
    :param rho_start: starting rho value in Hough space. When not defined by the user the starting value is 0
                     (optional, default=None).
    :param rho_stop: stopping rho value in Hough space. When not defined by the user the stopping value is defined based
                     on a point cloud. In default configuration the value will differ for each point cloud
                     (optional, default=None).
    :param point_cloud_cut_floor: (optional, default=None).
    :param point_cloud_cut_ceiling: (optional, default=None).
    """

    # check file or directory exists
    assert os.path.exists(path), "file or directory do not exist"
    # check whether single cloud or folder
    if os.path.isdir(path):
        # perform Hough Transform on a folder with .csv files
        run_hough_transform_folder(path, output_dir=output_dir, for_3d=for_3d, cut_cloud=cut_cloud,
                                   visualize_point_cloud=visualize_point_cloud,
                                   visualize_accumulator=visualize_accumulator, theta_step=theta_step, fi_step=fi_step,
                                   rho_step=rho_step, threshold=threshold, theta_start=theta_start,
                                   theta_stop=theta_stop, fi_start=fi_start, fi_stop=fi_stop, rho_start=rho_start,
                                   rho_stop=rho_stop, point_cloud_cut_floor=point_cloud_cut_floor,
                                   point_cloud_cut_ceiling=point_cloud_cut_ceiling)
    else:
        assert path.endswith('.csv'), "file format is wrong"
        # perform Hough Transform on a single point cloud
        run_hough_transform_single_cloud(path, output_dir=output_dir, for_3d=for_3d, cut_cloud=cut_cloud,
                                         visualize_point_cloud=visualize_point_cloud,
                                         visualize_accumulator=visualize_accumulator, theta_step=theta_step,
                                         fi_step=fi_step, rho_step=rho_step, threshold=threshold,
                                         theta_start=theta_start, theta_stop=theta_stop, fi_start=fi_start,
                                         fi_stop=fi_stop, rho_start=rho_start, rho_stop=rho_stop,
                                         point_cloud_cut_floor=point_cloud_cut_floor,
                                         point_cloud_cut_ceiling=point_cloud_cut_ceiling)


if __name__ == "__main__":

    run_hough_transform_examples_2d()
    run_hough_transform_examples_3d()
    # hough_transform("cube.csv", output_dir="altered_cube3d.csv", for_3d=True, cut_cloud=False,
    #                 visualize_point_cloud=True, visualize_accumulator=True, theta_step=0.01*math.pi,
    #                 fi_step=0.01 * math.pi, rho_step=0.1, threshold=0.05, rho_start=-15.0, rho_stop=15.0,
    #                 theta_start=-math.pi/2, theta_stop=math.pi/2, fi_start=-math.pi/2, fi_stop=math.pi/2)

    # hough_transform("rectangle3d.csv", output_dir="altered_rectangle3d.csv", for_3d=True, cut_cloud=False,
    #                 visualize_point_cloud=True, visualize_accumulator=True, theta_step=0.01*math.pi,
    #                 fi_step=0.01 * math.pi, rho_step=0.1, threshold=0.05, rho_start=-15.0, rho_stop=15.0,
    #                 theta_start=-math.pi/2, theta_stop=math.pi/2, fi_start=-math.pi/2, fi_stop=math.pi/2)
    #
    # hough_transform("pyramid3d.csv", output_dir="altered_pyramid3d.csv", for_3d=True, cut_cloud=False, visualize_point_cloud=True,
    #                 visualize_accumulator=True, theta_step=0.01*math.pi,
    #                 fi_step=0.01 * math.pi, rho_step=0.1, threshold=0.05, rho_start=-15.0, rho_stop=15.0,
    #                 theta_start=-math.pi/2, theta_stop=math.pi/2, fi_start=-math.pi/2, fi_stop=math.pi/2)
    #
    # hough_transform("corridor.csv", output_dir="altered_corridor3d.csv", for_3d=True, cut_cloud=False,
    #                 visualize_point_cloud=True, visualize_accumulator=True, theta_step=0.01*math.pi,
    #                 fi_step=0.01 * math.pi, rho_step=0.1, threshold=0.05, rho_start=-15.0, rho_stop=15.0,
    #                 theta_start=-math.pi/2, theta_stop=math.pi/2, fi_start=-math.pi/2, fi_stop=math.pi/2)

    run_hough_transform_real_data_examples()
