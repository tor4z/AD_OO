def dfs(in_node, visited):
    for node in in_node._input_nodes:
        if node not in visited:
            dfs(node, visited)
    visited.append(in_node)


def topsort(root):
    visited = []
    dfs(root, visited)
    return visited
