
import random, sys
INITIAL = [random.random() for _ in range(8)]
ORDER = list({'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf'})
HASH = hash('candidate import time')
def world_model_step(m,o,a): return (m or 0) + 1
def planner(m,o): return {'move':[random.choice([-1,0,1]),0], 'interact':False}
def export_model(m,o):
    return {'initial':INITIAL, 'order':ORDER, 'hash':HASH,
            'later':random.random(), 'paths':sys.path, 'steps':m}
