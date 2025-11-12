graph ={
    'A': {'B':2, 'E':3},
    'B' : {'C':11},
    'C':{'G':2},
    'E':{'D':60},
    'D':{'G':1},
    'G':{}
}

h = {'A':11, 'B': 6, 'C': 9, 'D': 1, 'E': 7, 'G': 0}

def a_star(start,goal):
    open_list = [start]
    came_from = {}
    g = {start:0}
    f = {start: h[start]}
    
    
    while open_list:
        current = min (open_list,key=lambda x:f[x])
        
        if current == goal:
            path = [current]
            
            while current in came_from:
                current = came_from[current]
                path.append(current)
                
            return list(reversed(path))
        
        
        open_list.remove(current)
        
        for neighbour, cost in graph[current].items():
            new_g = g[current]+cost
                
            if neighbour not in g or new_g <g[neighbour]:
                came_from[neighbour] = current
                g[neighbour]=new_g
                f[neighbour]=new_g + h[neighbour]
                
                if neighbour not in open_list:
                    open_list.append(neighbour)
    return None

path = a_star('A','G')
print("Shortest path:",path)