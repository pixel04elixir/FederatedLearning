import networkx as nx
import tensorflow as tf

from FLearning.fl_poison import FLPoison
from FLearning.utils import *


class FLSniper(FLPoison):
    def fl_cleanup(self, scaled_local_wt, thershold=0.2, cliques=5):
        edges = make_edges(scaled_local_wt, thershold)
        graph = {}

        for u, v in edges:
            if u not in graph:
                graph[u] = set()
            if v not in graph:
                graph[v] = set()

            graph[u].add(v)
            graph[v].add(u)

        max_clique = bron_kerbosch(nx.Graph(graph))
        if len(max_clique) <= cliques:
            raise Exception("Too few cliques to continue")

        return [scaled_local_wt[x] for x in max_clique]
