# /core/network.py

import networkx as nx
from core.node import Node
from core.edge import Edge
from core.sensor import Sensor
from core.user import UserNode


class Network:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node: Node):
        self.graph.add_node(node.id, node=node)

    def add_edge(self, edge: Edge):
        self.graph.add_edge(edge.source_node.id, edge.terminal_node.id, edge=edge)

    def get_node(self, node_id):
        return self.graph.nodes[node_id]['node'] if node_id in self.graph.nodes else None

    def get_edge(self, source_id, terminal_id):
        return self.graph.edges[source_id, terminal_id]['edge'] if self.graph.has_edge(source_id, terminal_id) else None

    def get_edge_nodes(self):
        nodes = []
        for _, data in self.graph.nodes(data=True):
            node = data['node']

            if not isinstance(node, (Sensor, UserNode)):
                nodes.append(node)
        return nodes

    def get_all_nodes(self):
        return [data['node'] for _, data in self.graph.nodes(data=True)]

    def get_all_edges(self):
        return [data['edge'] for _, _, data in self.graph.edges(data=True)]

    def get_path(self, source_id, target_id, data_size):
        def weight(u, v, d):
            edge = d['edge']
            return data_size / edge.bandwidth
        try:
            return nx.dijkstra_path(self.graph, source_id, target_id, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_path_bandwidth(self, path):
        bandwidths = []
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            bandwidths.append(edge.bandwidth)
        if len(path) <= 1:
            return float('inf')
        return min(bandwidths)  # bottleneck
