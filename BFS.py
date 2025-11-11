from collections import deque

def bfs(graph, start_node, goal_node):
    visited = []
    queue = deque([start_node])

    while queue:
        node = queue.popleft()

        if node not in visited:
            print(node, end=" ")
            visited.append(node)

            if node == goal_node:
                print("\nGoal Node Found:", goal_node)
                return

            for neighbour in graph[node]:
                if neighbour not in visited and neighbour not in queue:
                    queue.append(neighbour)

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

    bfs(graph, start, goal)
