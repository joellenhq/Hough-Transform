import numpy as np
import math


def calculate_center(*args):
    return [sum(tup)/len(args) for tup in zip(*args)]


def calculate_corrected_l_distance(len1, dist):
    n = round(dist/len1)
    corrected_l = dist/n
    return corrected_l


def create_points(p1, p2, n):
    # intercept theorem
    # unpack coordinates
    x1, y1 = p1
    x2, y2 = p2

    x_coor = np.linspace(x1, x2, n)
    y_coor = np.linspace(y1, y2, n)
    points = np.stack((x_coor, y_coor), axis=1)
    return points


calculate_distance = lambda p1, p2: math.sqrt(((p1[0] - p2[0])**2+(p1[1] - p2[1])**2))


def calculate_points(points, rearranged_points, len2):
    point_cloud_points = np.array([[points[0][0], points[0][1]]])
    for point1, point2 in zip(points, rearranged_points):
        distance = calculate_distance(point1, point2)
        n = round(distance/len2)
        points = create_points(point1, point2, n)
        point_cloud_points = np.concatenate((point_cloud_points, points), axis=0)
    return point_cloud_points


def calculate_points_for_line(p1, p2, len2):
    point_cloud_points = np.array([p1, p2])
    distance = calculate_distance(p1, p2)
    n = round(distance / len2)
    points = create_points(p1, p2, n)
    # print(point_cloud_points.shape)
    # print(points.shape)
    point_cloud_points = np.concatenate((point_cloud_points, points), axis=0)
    return point_cloud_points


def get_point_cloud(points, len4):
    rearranged_points = points.copy()
    rearranged_points.append(rearranged_points.pop(0))

    point_cloud_points = calculate_points(points, rearranged_points, len4)

    return point_cloud_points


def triangle_point_cloud(p1, p2, p3, len3):
    # unpack points
    points = [p1, p2, p3]
    # step_one = points.pop(0)
    # print(step_one)
    point_cloud_points = get_point_cloud(points, len3)
    point_cloud_points = add_uncertainty(point_cloud_points)
    save_points_to_csv("triangle2d.csv", point_cloud_points)


def square(a):
    center = (0, 0)
    p1 = tuple(np.subtract(center, (a/2, a/2)))
    p2 = tuple(np.subtract(center, (a/2, -a/2)))
    p3 = tuple(np.subtract(center, (-a/2, -a/2)))
    p4 = tuple(np.subtract(center, (-a/2, a/2)))
    return [p1, p2, p3, p4]


def rectangle(a, b):
    center = (0, 0)
    p1 = tuple(np.subtract(center, (a/2, b/2)))
    p2 = tuple(np.subtract(center, (a/2, -b/2)))
    p3 = tuple(np.subtract(center, (-a/2, -b/2)))
    p4 = tuple(np.subtract(center, (-a/2, b/2)))
    return [p1, p2, p3, p4]


def square_point_cloud(a, len3):
    points = square(a)
    point_cloud_points = get_point_cloud(points, len3)
    point_cloud_points = add_uncertainty(point_cloud_points)
    save_points_to_csv("square2d.csv", point_cloud_points)


def rectangle_point_cloud(a, b, len3):
    points = rectangle(a, b)
    point_cloud_points = get_point_cloud(points, len3)
    point_cloud_points = add_uncertainty(point_cloud_points)
    save_points_to_csv("rectangle2d.csv", point_cloud_points)


def save_points_to_csv(filename, point_cloud):
    np.savetxt(filename, point_cloud, delimiter=",")


def float_range(start, stop, inc):
    while start < stop:
        yield start
        start += inc


def rectangle_point_cloud_3d(a, b, c, len3):
    points = rectangle(a, b)
    point_cloud_2d_points = get_point_cloud(points, len3)
    # add Z dimention
    point_cloud_shape = point_cloud_2d_points.shape
    complete_point_cloud = np.array([[point_cloud_2d_points[0][0], point_cloud_2d_points[0][1], 0]])
    for z_value in float_range(0, c, len3):
        # add z dimention to existing array
        z = np.full(point_cloud_shape[0], z_value)
        z = np.reshape(z, (point_cloud_shape[0], 1))

        point_cloud_with_z = np.concatenate((point_cloud_2d_points, z), axis=1)
        complete_point_cloud = np.concatenate((complete_point_cloud, point_cloud_with_z))
    complete_point_cloud = add_uncertainty(complete_point_cloud)
    save_points_to_csv("rectangle3d.csv", complete_point_cloud)


def pyramid_point_cloud_3d(a, b, c, len3):
    complete_point_cloud = np.array([[0, 0, 0]])
    n = round(c/len3)
    len_a = a/n
    len_b = b/n
    for z_value in float_range(0, c, len3):
        points = rectangle(a, b)
        point_cloud_2d_points = get_point_cloud(points, len3)
        point_cloud_shape = point_cloud_2d_points.shape

        # add z dimention to existing array
        z = np.full(point_cloud_shape[0], z_value)
        z = np.reshape(z, (point_cloud_shape[0], 1))

        point_cloud_with_z = np.concatenate((point_cloud_2d_points, z), axis=1)
        complete_point_cloud = np.concatenate((complete_point_cloud, point_cloud_with_z))

        # change a and b
        a -= len_a
        b -= len_b
    complete_point_cloud = add_uncertainty(complete_point_cloud)
    save_points_to_csv("pyramid3d.csv", complete_point_cloud)


def plane_point_cloud_3d(a, b, len3):
    complete_point_cloud = np.array([[0, 0]])
    n = round(a/len3)
    len_a = a/n
    len_b = b/n
    points = rectangle(a, b)
    p1 = list(points[0])
    p2 = list(points[1])
    p3 = points[2]
    p4 = points[3]
    while p1[0] <= p4[0]:
        # print(p1)
        # print(p2)
        # print(p3)
        # print(p4)
        point_cloud_2d_points = calculate_points_for_line(p1, p2, len3)
        point_cloud_shape = point_cloud_2d_points.shape
        # print(point_cloud_shape)
        complete_point_cloud = np.concatenate((complete_point_cloud, point_cloud_2d_points))

        # # change a and b
        # a -= len_a
        # b -= len_b

        p1[0] += len_a
        p2[0] += len_a
    complete_point_cloud = add_uncertainty(complete_point_cloud)
    save_points_to_csv("plane.csv", complete_point_cloud)


def cube_point_cloud_3d(a, len3):
    points = square(a)
    point_cloud_2d_points = get_point_cloud(points, len3)
    # add Z dimention
    point_cloud_shape = point_cloud_2d_points.shape
    complete_point_cloud = np.array([[point_cloud_2d_points[0][0], point_cloud_2d_points[0][1], 0]])
    for z_value in float_range(0, a, len3):
        # add z dimention to existing array
        z = np.full(point_cloud_shape[0], z_value)
        z = np.reshape(z, (point_cloud_shape[0], 1))

        point_cloud_with_z = np.concatenate((point_cloud_2d_points, z), axis=1)
        complete_point_cloud = np.concatenate((complete_point_cloud, point_cloud_with_z))
    complete_point_cloud = add_uncertainty(complete_point_cloud)
    save_points_to_csv("cube.csv", complete_point_cloud)


def add_uncertainty(point_cloud, acc=0.015):
    random_array = np.random.normal(0, acc, size=point_cloud.shape)
    return point_cloud+random_array


if __name__ == "__main__":
    P1 = (-5, 2)
    P2 = (2, 3)
    P3 = (-1, 0.5)
    len0 = 0.05
    triangle_point_cloud(P1, P2, P3, len0)
    rectangle_point_cloud(1, 10, len0)
    square_point_cloud(10, len0)
    rectangle_point_cloud_3d(2, 3, 4, len0)
    cube_point_cloud_3d(10, len0)
    pyramid_point_cloud_3d(3, 6, 10, len0)
    plane_point_cloud_3d(3, 6, 0.01)

