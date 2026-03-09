from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.envs.avail_pos_helper import avail_pos_helper


class StoringFoodEnv(RoomGrid):
    """
    Environment in which the agent is instructed to clean a car
    """

    def __init__(
            self,
            mode='primitive',
            room_size=16,
            num_rows=1,
            num_cols=1,
            max_steps=1e5,
    ):
        num_objs = {'oatmeal': 2, 'countertop': 1, 'chip': 2, 'vegetable_oil': 2,
                    'sugar': 2, 'cabinet': 1}

        self.mission = 'store food'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        oatmeals = self.objs['oatmeal']
        chips = self.objs['chip']
        countertop = self.objs['countertop'][0]
        vegetable_oils = self.objs['vegetable_oil']
        sugars = self.objs['sugar']
        cabinet = self.objs['cabinet'][0]

        countertop.width, countertop.height = 4, 3

        self.place_obj(countertop)
        self.place_obj(cabinet)
        
        max_countertop_x, max_countertop_y = -1, -1
        min_countertop_x, min_countertop_y = 99, 99
        
        for pos_x, pos_y in countertop.all_pos:
            if pos_x > max_countertop_x:
                max_countertop_x = pos_x
            if pos_y > max_countertop_y:
                max_countertop_y = pos_y
            if pos_x < min_countertop_x:
                min_countertop_x = pos_x
            if pos_y < min_countertop_y:
                min_countertop_y = pos_y
        available_pos_list = [] 
        for pos_x, pos_y in countertop.all_pos:
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
                if 0 < n_dx < self.width and 0 < n_dy < self.height:
                    neighbours.append((n_dx, n_dy))
            for n_dx, n_dy in neighbours:
                first_item = self.grid.get_all_items(n_dx, n_dy)[0]
                if first_item is None:
                    has_space = True
                    break
            if not has_space:
                blocked_pos.append((pos_x, pos_y))
        for pos in blocked_pos:
            if pos in available_pos_list:
                available_pos_list.remove(pos)
                
        if len(available_pos_list) < 8:
            countertop_pos = self._rand_subset(available_pos_list, len(available_pos_list))
            self.put_obj(oatmeals[0], *countertop_pos[0], 1)
            self.put_obj(oatmeals[1], *countertop_pos[0], 2)
            self.put_obj(chips[0], *countertop_pos[1], 1)
            self.put_obj(chips[1], *countertop_pos[1], 2)
            self.put_obj(vegetable_oils[0], *countertop_pos[2], 1)
            self.put_obj(vegetable_oils[1], *countertop_pos[2], 2)
            self.put_obj(sugars[0], *countertop_pos[3], 1)
            self.put_obj(sugars[1], *countertop_pos[3], 2)
                
        else:
            
            countertop_pos = self._rand_subset(available_pos_list, 8)
            self.put_obj(oatmeals[0], *countertop_pos[0], 1)
            self.put_obj(oatmeals[1], *countertop_pos[1], 1)
            self.put_obj(chips[0], *countertop_pos[2], 1)
            self.put_obj(chips[1], *countertop_pos[3], 1)
            self.put_obj(vegetable_oils[0], *countertop_pos[4], 1)
            self.put_obj(vegetable_oils[1], *countertop_pos[5], 1)
            self.put_obj(sugars[0], *countertop_pos[6], 1)
            self.put_obj(sugars[1], *countertop_pos[7], 1)



    def _end_conditions(self):
        oatmeals = self.objs['oatmeal']
        chips = self.objs['chip']
        vegetable_oils = self.objs['vegetable_oil']
        sugars = self.objs['sugar']
        cabinet = self.objs['cabinet'][0]

        for obj in oatmeals + chips + vegetable_oils + sugars:
            if not obj.check_rel_state(self, cabinet, 'inside'):
                return False

        return True


# non human input env
register(
    id='MiniGrid-StoringFood-16x16-N2-v0',
    entry_point='mini_behavior.envs:StoringFoodEnv'
)

register(
    id='MiniGrid-StoringFood-12x12-N2-v0',
    entry_point='mini_behavior.envs:StoringFoodEnv',
    kwargs={'room_size': 12}
)

register(
    id='MiniGrid-StoringFood-11x11-N2-v0',
    entry_point='mini_behavior.envs:StoringFoodEnv',
    kwargs={'room_size': 11}
)

# human input env
register(
    id='MiniGrid-StoringFood-16x16-N2-v1',
    entry_point='mini_behavior.envs:StoringFoodEnv',
    kwargs={'mode': 'cartesian'}
)
