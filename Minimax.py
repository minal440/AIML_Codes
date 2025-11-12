def minimax(depth,max_depth,isMax,values,path):
    if depth==max_depth:
        return values.pop(0),path
    
    if isMax:
        left_val,left_path = minimax(depth+1,max_depth,False,values,path + ["Left"])
        right_val,right_path = minimax(depth+1,max_depth,False,values,path + ["Right"])
        
        if left_val > right_val:
            return left_val,left_path
        else:
            return right_val,right_path
        
    else:
        left_val,left_path = minimax(depth+1,max_depth,True,values,path + ["Left"])
        right_val,right_path = minimax(depth+1,max_depth,True,values,path + ["Right"])
        
        if left_val < right_val:
            return left_val,left_path
        else:
            return right_val,right_path
        


max_depth = int(input("Enter the depth\n"))

num_leaves = 2**max_depth

print(f"Enter { num_leaves} leaf node values\n")
values = list(map(int,input("Enter the leaf node values\n").split()))

if len(values)!=num_leaves:
    print("ERROR\n")
else:
    best_value, best_path = minimax(0,max_depth,True,values,[])
    print("\nBest Value Found:\n", best_value)
    print("Path Taken:", " -> ".join(best_path))