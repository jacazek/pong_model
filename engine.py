import numpy as np


# def random_velocity(max=0.025, min=0.005):
#     # return np.random.choice([np.random.uniform(min, max), np.random.uniform(max * -1, min * -1)])
#     random_ng = np.random.default_rng()
#     [x, y] = random_ng.uniform(max * -1, max, 2)
#     return x, y

def random_velocity_generator(min=0.005, max=0.025):
    # count = 0
    random_ng = np.random.default_rng()
    while True:
        x = random_ng.uniform(max*-1, min*-1) if random_ng.choice([True, False]) else random_ng.uniform(min, max)
        y = random_ng.uniform(max*-1, max)
        # random_x = np.random.uniform(min, max)
        # random_y = random_velocity(min=min, max=max)
        # count += 1
        yield x, y


