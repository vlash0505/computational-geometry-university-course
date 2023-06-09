import math
from matplotlib import pyplot as plt

import bintrees

#define DS region-------------------------------------------------------------------------------------------------------
class Point:
    def __init__(self, x, y):
        """
        Initializes the Point object with x, y coordinates and input, output weights.
        :param x: X coordinate
        :param y: Y coordinate
        """
        self.x = float(x)
        self.y = float(y)
        self.w_in = 0    # in weight
        self.w_out = 0   # out weight

    def __repr__(self):
        return "(" + str(self.x) + "; " + str(self.y) + ")"


class Edge:
    def __init__(self, start, end):
        """
        Initializes the Edge object with start and end points, weight, end x-coordinate(to), and rotation angle.
        :param start: Starting Point object
        :param end: Ending Point object
        """
        self.start = start
        self.end = end
        self.weight = 0
        self.to = end.x
        # Compute the rotation angle
        self.rotation = math.atan2(end.y - start.y, end.x - start.x)

#preprocess region------------------------------------------------------------------------------------------------------

def regularize_graph(vertices):
    """
    Regularize graph by creating new edges based on the proximity of vertices.
    :param vertices: list of Point objects representing vertices
    :return: list of regularized Edge objects
    """
    n = len(vertices)
    regularized_edges = []

    # Sort vertices based on y-coordinate
    vertices = sorted(vertices, key=lambda point: point.y)

    # Create new edges based on proximity of vertices
    for i in range(1, n - 1):
        if vertices[i].x != vertices[i - 1].x and vertices[i].x != vertices[i + 1].x:
            closest_point = vertices[i - 1] if abs(vertices[i].x - vertices[i - 1].x) < abs(
                vertices[i].x - vertices[i + 1].x) else vertices[i + 1]
            regularized_edges.append(Edge(vertices[i], closest_point))

    return regularized_edges


def weight_balancing_regular_PSLG(vertices, edges_in, edges_out):
    """
    Balances weight for a regular PSLG.
    :param vertices: list of vertices
    :param edges_in: list of edges coming in to the vertices
    :param edges_out: list of edges going out from the vertices
    """
    n = len(vertices)

    # Balancing from the bottom to the top
    for i in range(1, n - 1):
        vertices[i].w_in = calculate_weight(edges_in[i])
        vertices[i].w_out = calculate_weight(edges_out[i])
        edges_out[i] = sort_edges(edges_out[i])
        if vertices[i].w_in > vertices[i].w_out:
            edges_out[i][0].weight = vertices[i].w_in - vertices[i].w_out + 1

    # Balancing from the top to the bottom
    for i in range(n - 1, 1, -1):
        vertices[i].w_in = calculate_weight(edges_in[i])
        vertices[i].w_out = calculate_weight(edges_out[i])
        edges_in[i] = sort_edges(edges_in[i])
        if vertices[i].w_out > vertices[i].w_in:
            edges_in[i][0].weight = vertices[i].w_out - vertices[i].w_in + edges_in[i][0].weight

def read_data(file_name_vertices, file_name_edges):
    """
    Reads vertex and edge data from the provided files.
    :param file_name_vertices: file containing vertex data
    :param file_name_edges: file containing edge data
    :return: list of Point objects representing vertices and list of Edge objects representing edges
    """
    vertices = []
    edges = []
    edges_input_array = open(file_name_edges).read().split()
    vertices_input_array = open(file_name_vertices).read().split()

    # Read and create vertices
    for i in range(0, len(vertices_input_array), 2):
        x = int(vertices_input_array[i])
        y = int(vertices_input_array[i + 1])
        vertices.append(Point(x, y))

    # Regularize the graph
    regularized_edges = regularize_graph(vertices)
    edges.extend(regularized_edges)

    # Read and create edges
    for i in range(0, len(edges_input_array), 2):
        start_point = vertices[int(edges_input_array[i])]
        end_point = vertices[int(edges_input_array[i + 1])]
        edges.append(Edge(start_point, end_point))

    return vertices, edges

#processing and utils region--------------------------------------------------------------------------------------------

def calculate_weight(array):
    """
    Sums up the weights of the edges in the provided array.
    :param array: list of Edge objects
    :return: sum of weights
    """
    return sum(edge.weight for edge in array)


def sort_edges(array):
    """
    Sorts the edges in the provided array based on rotation.
    :param array: list of Edge objects
    :return: sorted list of Edge objects
    """
    return sorted(array, key=lambda edge: edge.rotation, reverse=True)


def create_chain(chain_num, ordered_edges_out, chains, vertices):
    """
    Creates a chain from the ordered edges.
    :param chain_num: chain number
    :param ordered_edges_out: list of Edge objects sorted by output
    :param chains: list of chains
    :param vertices: list of vertices
    """
    current_v = 0
    n = len(vertices)
    while current_v != n - 1:
        new_in_chain = get_leftmost_unused_edge(ordered_edges_out[current_v])
        chains[chain_num].append(new_in_chain)
        new_in_chain.weight -= 1
        current_v = vertices.index(new_in_chain.end)


def get_leftmost_edge(array):
    """
    Finds the leftmost edge from the given array.
    :param array: list of Edge objects
    :return: Edge object which is the leftmost in the array
    """
    array = sort_edges(array)
    return array[0]


def get_leftmost_unused_edge(array):
    """
    Finds the leftmost unused edge from the given PSLG.
    :param array: list of Edge objects
    :return: Edge object which is the leftmost unused in the array
    """
    for edge in array:
        if edge.weight > 0:
            return edge
    return None

#search region----------------------------------------------------------------------------------------------------------

def locate_point(point, chains, num_chains):
    """
    Find a point in chains.
    :param point: Point object to find
    :param chains: list of chains to search from
    :param num_chains: number of chains
    :return: String representing where the point is found
    """
    # Construct the balanced search tree
    bst = bintrees.AVLTree()

    # Insert the y-coordinates of the edges along with their corresponding chains
    for chain in chains:
        for edge in chain:
            bst.insert(edge.start.y, chain)

    for p in range(0, num_chains):
        for e in chains[p]:
            # Include points on the vertices or edges
            if e.start.y <= point.y <= e.end.y:
                point_vector = Point(point.x - e.start.x, point.y - e.start.y)
                edge_vector = Point(e.end.x - e.start.x, e.end.y - e.start.y)
                # Include points on the edge
                if math.atan2(point_vector.y, point_vector.x) >= math.atan2(edge_vector.y, edge_vector.x):
                    return "Point is between chains " + str(p - 1) + " , " + str(p)
    return "Point is not inside graph"


def plot(vertices, edges, point_to_locate):
    """
    Plots the graph using matplotlib.
    :param vertices: list of vertices
    :param edges: list of edges
    :param point_to_locate: point to locate
    """
    ax = plt.axes()

    for i, vertex in enumerate(vertices):
        plt.annotate(i, (vertex.x, vertex.y), fontsize=20)
        plt.plot(vertex.x, vertex.y, marker="o", markersize=4, markerfacecolor="green")

    for edge in edges:
        ax.arrow(edge.start.x, edge.start.y,
                 edge.end.x - edge.start.x, edge.end.y - edge.start.y, head_width=0, head_length=0)
        plt.annotate(edge.weight,
                     xy=((edge.end.x + edge.start.x) / 2, (edge.end.y + edge.start.y) / 2),
                     xytext=(10, -10),
                     textcoords='offset points',
                     fontsize=14)

    plt.plot(point_to_locate.x, point_to_locate.y, marker="o", markersize=8, markerfacecolor="red")
    plt.show()

#init region------------------------------------------------------------------------------------------------------------

def init():
    # Read the data from file
    vertices, edges = read_data("vertices.txt", "edges.txt")
    query_point = Point(9, 10)
    # Sort the vertices
    vertices = sorted(vertices, key=lambda point: point.y)

    edges_in = [[] for _ in vertices]
    edges_out = [[] for _ in vertices]

    # Assign edges to vertices
    for edge in edges:
        from_idx = vertices.index(edge.start)
        to_idx = vertices.index(edge.end)
        edges_out[from_idx].append(edge)
        edges_in[to_idx].append(edge)
        edge.weight = 1

    # Perform weight balancing
    weight_balancing_regular_PSLG(vertices, edges_in, edges_out)

    # Plot the graph
    plot(vertices, edges, query_point)

    # Create chains from the sorted edges
    chains = [[] for _ in range(calculate_weight(edges_out[0]))]
    ordered_edges_out = [sort_edges(v) for v in edges_out]

    for j in range(len(chains)):
        create_chain(j, ordered_edges_out, chains, vertices)

    # Print the chains
    for i, chain in enumerate(chains):
        print(f"Chain {i}: {vertices.index(chain[0].start)}", end="")
        for edge in chain:
            print(f" {vertices.index(edge.end)}", end="")
        print()

    # Find the point in the chains
    print(locate_point(query_point, chains, len(chains)))


if __name__ == "__main__":
    init()