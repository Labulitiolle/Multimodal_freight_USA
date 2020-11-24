"""This script was taken from his networkx repository and adapted for my use."""

from heapq import heappop, heappush
from itertools import count
from typing import Callable, Tuple
from geopy.distance import great_circle

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from networkx.algorithms.shortest_paths.weighted import _weight_function

__all__ = ["astar_path"]


def astar_path(G, source, target, heuristic=None, weight="weight"):
    """Returns a list of nodes in a shortest path between source and target
    using the A* ("A-star") algorithm.
    There may be more than one shortest path.  This returns only one.
    Parameters
    ----------
    G : NetworkX graph
    source : node
       Starting node for path
    target : node
       Ending node for path
    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.
    weight : string or function
       If this is a string, then edge weights will be accessed via the
       edge attribute with this key (that is, the weight of the edge
       joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
       such edge attribute exists, the weight of the edge is assumed to
       be one.
       If this is a function, the weight of an edge is the value
       returned by the function. The function must accept exactly three
       positional arguments: the two endpoints of an edge and the
       dictionary of edge attributes for that edge. The function must
       return a number.
    Raises

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    # The queue stores heuristic, priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path, explored

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            ncost = dist + weight(curnode, neighbor, w)
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

def heuristic_func(self, u: int, v: int) -> float:
    return great_circle(
        (self.G_multimodal_u.nodes[u]["y"], self.G_multimodal_u.nodes[u]["x"]),
        (self.G_multimodal_u.nodes[v]["y"], self.G_multimodal_u.nodes[v]["x"]),
    ).miles

def plot_astar_exploration(
    G,
    orig: Tuple[float, float],
    dest: Tuple[float, float],
    target_weight: str = "CO2_eq_kg",
    heuristic: Callable[[int, int], float] = None,
    orig_dest_size: int = 100,
):
    if not heuristic:
        heuristic = heuristic_func()

    node_orig, dist_orig = ox.get_nearest_node(
        G, orig, method="haversine", return_dist=True
    )
    node_dest, dist_dest = ox.get_nearest_node(
        G, dest, method="haversine", return_dist=True
    )

    dist_path = nx.astar_path_length(
        G, node_orig, node_dest, weight="dist_miles"
    )
    weight_path = nx.astar_path_length(
        G, node_orig, node_dest, weight=target_weight
    )

    shortest_path_nodes, explored = astar_path(
        G,
        node_orig,
        node_dest,
        weight=target_weight,
        heuristic=heuristic,
    )

    distance_highway = dist_path

    dist_non_highway = dist_orig + dist_dest

    total_dist = dist_non_highway + distance_highway

    explored_nodes = set(explored.keys())
    explored_nodes.update(explored.values())
    explored_nodes.remove(None)

    nodes = self.nodes.copy()
    explored_nodes = nodes.loc[list(explored_nodes), :]

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.scatter(orig[1], orig[0], marker="x", s=orig_dest_size, zorder=5)
    ax.scatter(dest[1], dest[0], marker="x", s=orig_dest_size, zorder=10)
    ax.scatter(
        explored_nodes.x, explored_nodes.y, marker="o", color="red", s=2, zorder=100
    )

    ox.plot_graph(
        G,
        edge_color="#bfbfbf",
        node_color="#595959",
        bgcolor="w",
        ax=ax,
        show=False,
    )
    ox.plot_graph_route(
        G,
        shortest_path_nodes,
        route_color="g",
        route_linewidth=4,
        route_alpha=0.9,
        orig_dest_size=100,
        ax=ax,
    )

    self.print_graph_info(total_dist, weight_path, target_weight)
