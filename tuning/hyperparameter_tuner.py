from python_visualisation import main_app
from multiprocessing import Pool, Queue, Manager
from typing import Optional
import global_settings as gs
import itertools as it
import pickle
import os
from misc_funcs import stdfrm

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


TUNING_SETTINGS = {
    "LEARNING_RATE": [stdfrm(1, -5), stdfrm(5, -5)],
    "TARGET_NET_COPY_STEPS": [3000, 10000],
    "DISCOUNT_RATE": [0.93, 0.98]
}


class Results:
    # stores results for every process
    # TODO: currently, when loaded, a process manager isnt used on the queue.
    #  This means it cannot be unpickled and used in multiprocessing.
    def __init__(self, m: Optional[Manager]):
        if m is None:
            self.q = Queue()
        else:
            self.q = m.Queue()

    def __iter__(self):
        # return queue as iterable
        objs = []
        for i in range(self.q.qsize()):
            obj = self.q.get()
            objs.append(obj)
            self.q.put(obj)

        return iter(objs)

    def save(self, name):
        with open("tuning/tuning_results/"+name, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        with open("tuning/tuning_results/" + name, "rb") as file:
            results = pickle.load(file)
        return results

    def __getstate__(self):
        return self.__iter__()

    def __setstate__(self, state):
        self.__init__(None)
        for obj in state:
            self.q.put(obj)


def process_func(params):
    print("STARTING...", params[:-1])
    queue = params[-1]
    for name, param in zip(TUNING_SETTINGS.keys(), params[:-1]):
        gs.Q_LEARNING_SETTINGS[name] = param
    q_learning_model = main_app(map_override=r"10.09;00.53", background=True, end_at_min_epsilon=True, verbose=0)
    print("COMPLETED", params[:-1])
    queue.put(q_learning_model)


if __name__ == '__main__':
    # move to parent directory so all file references work
    os.chdir("../")

    m = Manager()
    results = Results(m)

    args = list(it.product(*TUNING_SETTINGS.values()))
    # add queue to all args
    args_with_q = [params + (results.q,) for params in args]

    print("MAIN PID:", os.getpid())
    with Pool(12) as pool:
        pool.map(process_func, args_with_q)
    results.save("nn 10 layers up to 96")
