def avail_pos_helper(furniture, env):
    max_countertop_x, max_countertop_y = -1, -1
    min_countertop_x, min_countertop_y = 99, 99
    
    for pos_x, pos_y in furniture.all_pos:
        if pos_x > max_countertop_x:
            max_countertop_x = pos_x
        if pos_y > max_countertop_y:
            max_countertop_y = pos_y
        if pos_x < min_countertop_x:
            min_countertop_x = pos_x
        if pos_y < min_countertop_y:
            min_countertop_y = pos_y
    available_pos_list = [] 
    for pos_x, pos_y in furniture.all_pos:
        if (pos_x in [min_countertop_x, max_countertop_x] or
            pos_y in [min_countertop_y, max_countertop_y]):
            available_pos_list.append((pos_x, pos_y))
    # continue to remove available position if its neighbour is occupied
    blocked_pos = []
    for pos_x, pos_y in available_pos_list:
        neighbours = []
        has_space = False
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            n_dx, n_dy = pos_x + dx, pos_y + dy
            if 0 < n_dx < env.width and 0 < n_dy < env.height:
                neighbours.append((n_dx, n_dy))
        for n_dx, n_dy in neighbours:
            first_item = env.grid.get_all_items(n_dx, n_dy)[0]
            if first_item is None:
                has_space = True
                break
        if not has_space:
            blocked_pos.append((pos_x, pos_y))
    for pos in blocked_pos:
        if pos in available_pos_list:
            available_pos_list.remove(pos)
            
    return available_pos_list