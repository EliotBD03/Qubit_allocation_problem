import numpy as np

n = 10
m = 15

def cross_solutions(layout1, layout2):
    """
    Will cross two solutions to get 2 children
    The child with the best cost will be returned
    """
    random_index = np.random.choice(n, n//2, replace=False)
    complement_index = [i for i in range(n) if i not in random_index]
    
    child1 = np.array([None] * n)
    child1[random_index] = layout1[random_index]
    
    child2 = np.array([None] * n)
    child2[random_index] = layout2[random_index]

    complement_part_child1 = layout2[complement_index]
    complement_part_child2 = layout1[complement_index]
    
    for index in complement_index:
        i = 0
        while i < complement_part_child1.size and complement_part_child1[i] in child1: i += 1
        if i == complement_part_child1.size:
            random_nb = np.random.randint(0, m - 1)
            while random_nb in child1 : random_nb = np.random.randint(0, m - 1)
            child1[index] = random_nb
        else:
            child1[index] = complement_part_child1[i]

        i = 0
        while i < complement_part_child2.size and complement_part_child2[i] in child2: i += 1
        if i == complement_part_child2.size:
            random_nb = np.random.randint(0, m - 1)
            while random_nb in child2 : random_nb = np.random.randint(0, m - 1)
            child2[index] = random_nb
        else:
            child2[index] = complement_part_child2[i]
    
   
    return child1, child2
        





if __name__ == "__main__":
    layouts = [np.random.choice(m,n, replace=False) for _ in range(n)]
    print(layouts)
    print()
    crossed_best_layouts = [[] for _ in range(len(layouts))]
    blacklist = [[]] * len(crossed_best_layouts)
    for i in range(len(crossed_best_layouts)):
        random_index = np.random.randint(0,len(crossed_best_layouts) - 1)
        while i != random_index and i in blacklist[random_index] : random_index = np.random.randint(0, len(crossed_best_layouts) - 1)
        print(f"parent one : {layouts[i]}, parent two : {layouts[random_index]}")
        crossed_best_layouts[i] = cross_solutions(layouts[i], layouts[random_index])
        print(crossed_best_layouts[i])
        blacklist.append(i)
