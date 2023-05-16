import networkx as nx

from FLearning.fl_poison import FLPoison
from FLearning.utils import *


class FLSniper(FLPoison):
    def fl_cleanup(self, scaled_local_wt, thershold=0.2, epsilon=0.05, cliques=5):
        while True:
            edges = make_edges(scaled_local_wt, thershold)
            graph = {}

            for u, v in edges:
                if u not in graph:
                    graph[u] = set()
                if v not in graph:
                    graph[v] = set()

                graph[u].add(v)
                graph[v].add(u)

            max_clique = get_max_clique(graph)
            if len(max_clique) > cliques:
                break

            thershold += epsilon

        print(max_clique)
        return max_clique
