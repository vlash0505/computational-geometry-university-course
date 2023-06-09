import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

class Node:
    """
    A Node class for use in a binary search tree (BST).

    Parameters:
        value : int
            The value of the node.

    Attributes:
        value : int
            The value of the node.
        left : Node
            The left child node.
        right : Node
            The right child node.
    """
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class SweepLine:
    """
    A binary search tree for representing a sweep line.

    Parameters:
        None

    Attributes:
        root : Node
            The root of the binary search tree.
    """
    def __init__(self):
        self.root = None

    def insert(self, value):
        """
        Inserts a new value into the binary search tree.

        Parameters:
            value : int
                The value to be inserted.

        Returns:
            None
        """
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        # Insert a value recursively in the BST
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
        else:
            raise ValueError("Value already exists in the binary search tree.")

    def delete(self, value):
        """
        Deletes a value from the binary search tree.

        Parameters:
            value : int
                The value to be deleted.

        Returns:
            None
        """
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        # Delete a node recursively from the BST
        if node is None:
            return None

        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                min_value = self._find_min_value(node.right)
                node.value = min_value
                node.right = self._delete_recursive(node.right, min_value)

        return node

    def _find_min_value(self, node):
        # Find minimum value in a given subtree
        current = node
        while current.left is not None:
            current = current.left
        return current.value

    def find_neighbours(self, value):
        """
        Finds the left and right neighbour of a value in the binary search tree.

        Parameters:
            value : int
                The value whose neighbours need to be found.

        Returns:
            left : int or None
                The left neighbour of the value.
            right : int or None
                The right neighbour of the value.
        """
        left = None
        right = None
        current = self.root

        while current is not None:
            if value < current.value:
                right = current
                current = current.left
            elif value > current.value:
                left = current
                current = current.right
            else:
                break

        return left.value if left else None, right.value if right else None


def cross_product(x1, y1, x2, y2):
    """
    Computes the cross product of two 2D vectors.

    Parameters:
        x1, y1 : float
            The x and y components of the first vector.
        x2, y2 : float
            The x and y components of the second vector.

    Returns:
        float
            The cross product of the two vectors.
    """
    return x1 * y2 - y1 * x2


def intersect(o1, p1, o2, p2):
    """
    Checks if two line segments (o1, p1) and (o2, p2) intersect.

    Parameters:
        o1, p1, o2, p2 : tuple of float
            The end points of the two line segments.

    Returns:
        bool
            True if the line segments intersect, False otherwise.
    """
    d1 = (p1[0] - o1[0], p1[1] - o1[1])
    d2 = (p2[0] - o2[0], p2[1] - o2[1])

    cross = cross_product(d1[0], d1[1], d2[0], d2[1])

    x = (o2[0] - o1[0], o2[1] - o1[1])

    if abs(cross) == 0:
        return False

    t = cross_product(x[0], x[1], d2[0], d2[1])
    u = cross_product(x[0], x[1], d1[0], d1[1])

    t = float(t) / cross
    u = float(u) / cross

    if 0 < t < 1 and 0 < u < 1:
        return True

    return False


def find_intersections(segments):
    """
    Finds all intersecting line segments.

    Parameters:
        segments : list of tuples
            The list of line segments represented as tuples of their end points.

    Returns:
        list of list of tuples
            The list of intersecting line segments.
    """
    end_points = []
    for i, (x1, _, x2, _) in enumerate(segments):
        end_points.append((x1, i, x1 >= x2))
        end_points.append((x2, i, x1 < x2))

    end_points = sorted(end_points)
    sweep_line = SweepLine()
    res = []

    for _, label, is_right in end_points:
        segment = segments[label]
        if not is_right:
            sweep_line.insert(label)
            for n in sweep_line.find_neighbours(label):
                if n is not None and intersect(
                        (segment[0], segment[1]),
                        (segment[2], segment[3]),
                        (segments[n][0], segments[n][1]),
                        (segments[n][2], segments[n][3])
                ):
                    res.append([segment, segments[n]])
        else:
            p, s = sweep_line.find_neighbours(label)
            if p is not None and s is not None:
                predecessor = segments[p]
                successor = segments[s]
                if intersect((predecessor[0], predecessor[1]), (predecessor[2], predecessor[3]),
                             (successor[0], successor[1]), (successor[2], successor[3])):
                    res.append([predecessor, successor])
            sweep_line.delete(label)

    return res


def calculate_intersection(x11, y11, x12, y12, x21, y21, x22, y22):
    """
    Calculates the intersection point of two line segments.

    Parameters:
        x11, y11, x12, y12, x21, y21, x22, y22 : float
            The coordinates of the end points of the two line segments.

    Returns:
        tuple of float or None
            The coordinates of the intersection point, or None if the line segments do not intersect.
    """
    # Calculate the intersection point of two line segments
    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21

    denominator = dx1 * dy2 - dy1 * dx2

    if denominator == 0:
        # The two lines are parallel or coincident
        return None

    ua = (dx2 * (y11 - y21) - dy2 * (x11 - x21)) / denominator
    ub = (dx1 * (y11 - y21) - dy1 * (x11 - x21)) / denominator

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Intersection exists within the line segments
        intersection_x = x11 + ua * dx1
        intersection_y = y11 + ua * dy1
        return intersection_x, intersection_y
    else:
        # Intersection exists but not within the line segments
        return None


def plot_line_segments(segments, intersections=None):
    """
    Plots the line segments and their intersections.

    Parameters:
        segments : numpy array
            The line segments to be plotted.
        intersections : list of tuple, optional
            The intersection points to be plotted.

    Returns:
        None
    """
    lines = LineCollection(segments)
    fig, ax = plt.subplots()
    ax.add_collection(lines)
    ax.autoscale()
    ax.margins(0.1)

    if intersections:
        for segment, intersection in zip(segments, intersections):
            intersection_x, intersection_y = intersection
            ax.plot(intersection_x, intersection_y, 'ro')
            ax.plot([segment[0, 0], segment[1, 0]], [segment[0, 1], segment[1, 1]], 'b-')

    else:
        for segment in segments:
            ax.plot([segment[0, 0], segment[1, 0]], [segment[0, 1], segment[1, 1]], 'b-')

    plt.show()


# Read line segments from a file and find their intersections
with open('input.txt', 'r') as file:
    N = file.readline()
    lines = [list(map(int, x.split(' '))) for x in file.readlines()]

result = find_intersections(lines)

intersections = []
for (x11, y11, x12, y12), (x21, y21, x22, y22) in result:
    segments = np.array([[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)]])
    intersection = calculate_intersection(x11, y11, x12, y12, x21, y21, x22, y22)
    intersections.append(intersection)  # Append the intersection, even if it's None

segments = np.array(lines).reshape(-1, 2, 2)
plot_line_segments(segments, intersections)

print("All line segments:")
for line_segment in lines:
    print(line_segment)
