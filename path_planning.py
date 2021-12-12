import sys
import math
import cv2 as cv
import numpy as np
from numpy.linalg import norm

class Point:
    def __init__(self, newX, newY):
        self.x = newX
        self.y = newY

def is_common_point(seg1, seg2):
    for point in seg1:
        if point in seg1 :
            return True
    return False

def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

def ccw2(A,B,C):
    return (C.y-A.y) * (B.x-A.x) >= (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return (ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)) and (ccw2(A,C,D) != ccw2(B,C,D) and ccw2(A,B,C) != ccw2(A,B,D))

def is_intersection(seg1, seg2):
    p1 = Point(seg1[0][0], seg1[0][1])
    p2 = Point(seg1[1][0], seg1[1][1])
    p3 = Point(seg2[0][0], seg2[0][1])
    p4 = Point(seg2[1][0], seg2[1][1])
    return intersect(p1, p2, p3, p4)

def is_same_segment(seg1, seg2):
    seg1 = np.array(seg1)
    seg2 = np.array(seg2)
    return (np.array_equal(seg1, seg2) or np.array_equal(seg1, seg2[[1, 0]]))

def is_edge_crossing(segment, obstacles):
    obstacle_nodes = []
    obstacle_edges = []
    for poly in obstacles:
        obstacle_nodes = obstacle_nodes + poly.vertices
        obstacle_edges = obstacle_edges + poly.edges
    for edge in obstacle_edges:
        seg2 = [obstacle_nodes[edge[0]], obstacle_nodes[edge[1]]]
        if is_intersection(segment, seg2):
            return True
    return False

def is_obstacle_edge(segment, obstacles):
    obstacle_nodes = []
    obstacle_edges = []
    for poly in obstacles:
        obstacle_nodes = obstacle_nodes + poly.vertices
        obstacle_edges = obstacle_edges + poly.edges
    for edge in obstacle_edges:
        seg2 = [obstacle_nodes[edge[0]], obstacle_nodes[edge[1]]]
        if is_same_segment(segment, seg2):
            return True
    return False

def is_point_inside_obstacle(point, poly):
    sign = []
    for i in range(len(poly.vertices)-1):
        edge = np.array(poly.vertices[i+1])-np.array(poly.vertices[i])
        v = point-np.array(poly.vertices[i])
        sign.append(np.cross(edge, v))
    return all([s >= 0 for s in sign]) or all([s <= 0 for s in sign])

def is_seg_inside_obstacle(seg, obstacles):
    if is_obstacle_edge(seg, obstacles):
        return False
    p1 = np.array(seg[0])
    p2 = np.array(seg[1])
    for poly in obstacles:
        if is_point_inside_obstacle(p1, poly) and is_point_inside_obstacle(p2, poly):
            return True
    return False

def get_graph_features(start, obstacles, goal):
    nodes = []
    edges = [] 
    for poly in obstacles:
        nodes = nodes + poly.vertices
    nodes.append(goal)
    nodes.append(start)
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            seg = [nodes[i], nodes[j]]
            if is_edge_crossing(seg, obstacles):
                continue
            if is_seg_inside_obstacle(seg, obstacles):
                continue
            else:
                edges.append((i, j))
    return nodes, edges

# Taken from https://www.udacity.com/blog/2021/10/implementing-dijkstras-algorithm-in-python.html
class Graph(object):    
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
        
    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]

def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
    
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
    
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
    
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
        
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
        
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path

def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
        
    # Add the start node manually
    path.append(start_node)
    
    # Display result :
    # print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    # print([nodes for nodes in reversed(path)])
    return [nodes for nodes in reversed(path)]

def graph_initialisation(start, obstacles, goal):
    node_coords, edges = get_graph_features(start, obstacles, goal)
    node_idx = [i for i in range(len(node_coords))]
    
    init_graph = {}
    for idx in node_idx:
        init_graph[idx] = {}
        
    for index1,index2 in edges:
        init_graph[index1][index2] = norm(np.array(node_coords[index1]) - np.array(node_coords[index2]))
    return Graph(node_idx, init_graph), node_idx, node_coords

def draw_graph(image, nodes, edges):
    # Draw visibility graph :
    for edge in edges :
        cv.line(image, nodes[edge[0]], nodes[edge[1]], (255,0,0), 2)
    return image

def draw_path(image, path_coords):
    # Draw path :
    for i in range(len(path_coords)-1):
        cv.line(image, path_coords[i], path_coords[i+1], (0,0,255), 2)
    return image

def find_path(start, obstacles, goal):
    graph, node_idx, node_coords = graph_initialisation(start, obstacles, goal)
    previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=node_idx[-1])
    shortest_path = print_result(previous_nodes, shortest_path, start_node=node_idx[-1], target_node=node_idx[-2])
    path_coords = [node_coords[idx] for idx in shortest_path]
    return path_coords
