from enum import IntEnum
from matplotlib import pyplot as plt
from matplotlib import patches as patches


class Dim(IntEnum):
    """
    Dimension Enum for VERTICAL and HORIZONTAL values
    """
    VERTICAL = 0
    HORIZONTAL = 1

class TreeNode:
    """
    TreeNode class represents the nodes in the Range Tree
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.point_index = None
        self.point = None
        self.line_dim = None


def _sorted_list_split(left_list, right_list, to_split_list):
    """
    Function to split the list into left and right sublists
    :param left_list: list of left points
    :param right_list: list of right points
    :param to_split_list: list to be split
    :return: Two lists after splitting
    """
    if not to_split_list:
        return [], []

    sep_flags = [2] * (max(to_split_list) + 1)
    for p in left_list:
        sep_flags[p] = 0
    for p in right_list:
        if sep_flags[p] != 2:
            raise Exception("_sorted_list_split")
        sep_flags[p] = 1

    split = [[], [], []]
    for p in to_split_list:
        split[sep_flags[p]].append(p)
    return split[0], split[1]


def _m(node):
    """
    Return the value at the current line dimension for a node
    :param node: TreeNode
    :return: Point value at line dimension
    """
    return node.point[node.line_dim]


def _is_inside_rect(point, x_range, y_range):
    """
    Function to check if a point lies inside a given rectangle
    :param point: Point to check
    :param x_range: Range on x-axis
    :param y_range: Range on y-axis
    :return: Boolean if point lies inside the rectangle
    """
    return (x_range[0] <= point[0] <= x_range[1]) and (y_range[0] <= point[1] <= y_range[1])


def construct_range_tree(points, dim_points, non_dim_points, dim):
    """
    Recursive function to construct the range tree
    :param points: List of points
    :param dim_points: List of points in current dimension
    :param non_dim_points: List of points not in current dimension
    :param dim: Current dimension
    :return: Root node of the Range Tree
    """
    if not dim_points:
        return None

    m_index = (len(dim_points) - 1) // 2
    m = dim_points[m_index]

    left_dim_points, right_dim_points = dim_points[:m_index], dim_points[m_index + 1:]
    left_non_dim_points, right_non_dim_points = _sorted_list_split(left_dim_points, right_dim_points, non_dim_points)

    next_dim = Dim.HORIZONTAL if dim == Dim.VERTICAL else Dim.VERTICAL

    node = TreeNode()
    node.point_index = m
    node.point = points[m]
    node.line_dim = dim

    node.left = construct_range_tree(points, left_non_dim_points, left_dim_points, next_dim)
    node.right = construct_range_tree(points, right_non_dim_points, right_dim_points, next_dim)

    return node


def preprocessing(points):
    """
    Preprocess the points to construct the range tree
    :param points: List of points
    :return: Root of the Range Tree
    """
    x = y = list(range(len(points)))
    x = sorted(x, key=lambda i: points[i][0])
    y = sorted(y, key=lambda i: points[i][1])
    return construct_range_tree(points, x, y, Dim.VERTICAL)


def _range_search(node, x_range, y_range, res):
    """
    Recursive function to perform the range search on the range tree
    :param node: Current node
    :param x_range: Range on x-axis
    :param y_range: Range on y-axis
    :param res: Result list to store the points inside the range
    """
    left, right = x_range if node.line_dim == Dim.VERTICAL else y_range
    m = _m(node)

    if left <= m <= right and _is_inside_rect(node.point, x_range, y_range):
        res.append([node.point_index, node.point])

    if node.left and left < m:
        _range_search(node.left, x_range, y_range, res)

    if node.right and m < right:
        _range_search(node.right, x_range, y_range, res)


def range_search(tree, x_range, y_range):
    """
    Perform range search on the given range tree
    :param tree: Root of the Range Tree
    :param x_range: Range on x-axis
    :param y_range: Range on y-axis
    :return: List of points inside the given range
    """
    res = []
    _range_search(tree, x_range, y_range, res)
    return res


def read_points(filename):
    """
    Function to read the points from a file
    :param filename: Name of the file
    :return: List of points
    """
    points = []
    with open(filename) as f:
        for line in f:
            x, y = map(float, line.split())
            points.append((x, y))
    return points


def read_region(filename):
    """
    Function to read the region from a file
    :param filename: Name of the file
    :return: Range on x-axis and y-axis
    """
    with open(filename) as f:
        x = list(map(float, f.readline().split()))
        y = list(map(float, f.readline().split()))
    return x, y


def init():
    """
    Initialize and start the process
    """
    points = read_points("points.txt")
    x_region, y_region = read_region("regions.txt")

    # Prepare the plots
    fig, ax = plt.subplots(2)
    for i in range(2):
        ax[i].set_xlim([0, 30])
        ax[i].set_ylim([0, 30])

        # Draw the rectangular region
        rect = patches.Rectangle((x_region[0], y_region[0]), x_region[1] - x_region[0], y_region[1] - y_region[0], linewidth=1, edgecolor='b', facecolor='none')
        ax[i].add_patch(rect)

    # Plot the points
    for point in points:
        ax[0].add_patch(patches.Circle(point, radius=0.051, color='b'))

    # Build the range tree and perform range search
    tree = preprocessing(points)
    result = range_search(tree, x_region, y_region)

    # Plot the points inside the rectangular region
    for point in result:
        ax[1].add_patch(patches.Circle((point[1][0], point[1][1]), radius=0.051, color='b'))

    plt.show()


# Run the program
if __name__ == "__main__":
    init()
