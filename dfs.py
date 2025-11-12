from collections import deque

def dfs(graph, start_node, goal_node):
    visited = []
    stack = [start_node]   # stack instead of queue

    while stack:
        node = stack.pop()  # pop() removes from the END (LIFO order)

        if node not in visited:
            print(node, end=" ")
            visited.append(node)

            if node == goal_node:
                print("\nGoal Node Found:", goal_node)
                return

            # Add neighbors to the stack (reversed to keep left-to-right order)
            for neighbour in reversed(graph[node]):
                if neighbour not in visited and neighbour not in stack:
                    stack.append(neighbour)

    print("\nGoal Node NOT Found in Graph.")


if __name__ == "__main__":
    graph = {
        'A': ['B', 'C', 'D'],
        'B': ['E', 'F'],
        'C': [],
        'D': ['L', 'M', 'N'],
        'E': [],
        'F': ['G', 'H'],
        'G': [],
        'H': [],
        'L': [],
        'M': [],
        'N': []
    }

    start = 'A'
    goal = 'H'

    print("Graph:", graph)
    print("Start:", start)
    print("Goal:", goal)
    print("--------------------------")

    dfs(graph, start, goal)
