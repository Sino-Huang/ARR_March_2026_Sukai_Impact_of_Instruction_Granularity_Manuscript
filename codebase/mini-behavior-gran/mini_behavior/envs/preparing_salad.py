from mini_behavior.roomgrid import *
from mini_behavior.register import register
from mini_behavior.envs.avail_pos_helper import avail_pos_helper


class PreparingSaladEnv(RoomGrid):
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
        num_objs = {'electric_refrigerator': 1, 'lettuce': 1, 'countertop': 1, 'apple': 1, 'tomato': 1,
                    'radish': 1, 'carving_knife': 1, 'plate': 2, 'cabinet': 1, 'sink': 1}

        self.mission = 'prepare salad'

        super().__init__(mode=mode,
                         num_objs=num_objs,
                         room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=max_steps
                         )

    def _gen_objs(self):
        electric_refrigerator = self.objs['electric_refrigerator'][0]
        lettuce = self.objs['lettuce']
        countertop = self.objs['countertop'][0]
        apple = self.objs['apple']
        tomato = self.objs['tomato']
        radish = self.objs['radish']
        carving_knife = self.objs['carving_knife'][0]
        plate = self.objs['plate']
        cabinet = self.objs['cabinet'][0]
        sink = self.objs['sink'][0]

        self.place_obj(countertop)
        self.place_obj(electric_refrigerator)
        self.place_obj(cabinet)
        self.place_obj(sink)

        countertop_pos = self._rand_subset(avail_pos_helper(countertop, self), 6)
        self.put_obj(lettuce[0], *countertop_pos[0], 1)
        # self.put_obj(lettuce[1], *countertop_pos[1], 1)
        self.put_obj(apple[0], *countertop_pos[2], 1)
        # self.put_obj(apple[1], *countertop_pos[3], 1)

        fridge_pos = self._rand_subset(avail_pos_helper(electric_refrigerator, self), 2)
        self.put_obj(tomato[0], *fridge_pos[0], 0)
        tomato[0].states['inside'].set_value(electric_refrigerator, True) # set tomato inside the fridge
        # self.put_obj(tomato[1], *fridge_pos[1], 2)

        self.put_obj(radish[0], *countertop_pos[4], 1)
        # self.put_obj(radish[1], *countertop_pos[5], 1)

        cabinet_pos = self._rand_subset(avail_pos_helper(cabinet, self), 3)
        self.put_obj(plate[0], *cabinet_pos[0], 0)
        plate[0].states['dustyable'].set_value(False)
        plate[0].states['inside'].set_value(cabinet, True)
        self.put_obj(plate[1], *cabinet_pos[1], 0)
        plate[1].states['dustyable'].set_value(False)
        plate[1].states['inside'].set_value(cabinet, True)
        self.put_obj(carving_knife, *cabinet_pos[2], 0)
        carving_knife.states['inside'].set_value(cabinet, True)
        
        # set not sliceable at the beginning
        for obj in apple + tomato:
            obj.states['sliceable'].set_value(False)



    def _end_conditions(self):
        lettuces = self.objs['lettuce']
        apples = self.objs['apple']
        tomatos = self.objs['tomato']
        radishes = self.objs['radish']
        plates = self.objs['plate']

        # Check that all vegetables are on plates
        all_vegetables = lettuces + apples + tomatos + radishes
        for veg in all_vegetables:
            # print(f"Checking {veg.name} on plates...")
            on_plate = False
            for plate in plates:
                if veg.check_rel_state(self, plate, 'onTop'):
                    on_plate = True
                    break
            if not on_plate:
                # print("find all vegetables on plates failed")
                return False

        # Check that apples and tomatoes are sliced
        for apple in apples:
            if not apple.check_abs_state(self, 'sliceable'):
                # print("find all apples sliced failed")
                return False
        
        for tomato in tomatos:
            if not tomato.check_abs_state(self, 'sliceable'):
                # print("find all tomatoes sliced failed")
                return False

        # Check that all plates have at least one vegetable on them
        for plate in plates:
            has_vegetable = False
            for veg in all_vegetables:
                if veg.check_rel_state(self, plate, 'onTop'):
                    has_vegetable = True
                    break
            if not has_vegetable:
                # print("find all plates with vegetables failed")
                return False

        return True


# non human input env
register(
    id='MiniGrid-PreparingSalad-16x16-N2-v0',
    entry_point='mini_behavior.envs:PreparingSaladEnv'
)

register(
    id='MiniGrid-PreparingSalad-12x12-N2-v0',
    entry_point='mini_behavior.envs:PreparingSaladEnv',
    kwargs={'room_size': 12}
)


# human input env
register(
    id='MiniGrid-PreparingSalad-16x16-N2-v1',
    entry_point='mini_behavior.envs:PreparingSaladEnv',
    kwargs={'mode': 'cartesian'}
)
