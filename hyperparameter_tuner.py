from python_visualisation import main_app
from multiprocessing import Pool, Queue
import global_settings as gs
import itertools as it
import pickle

TUNING_SETTINGS = {
    "LEARNING_RATE": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
    "TARGET_NET_COPY_STEPS": [5000, 20000, 50000, 100000]
}


class Results:
    def __init__(self):
        self.q = Queue()

    def __iter__(self):
        # return queue as list
        objs = []
        for i in range(self.q.qsize()):
            obj = self.q.get()
            objs.append(obj)
            self.q.put(obj)

        return iter(objs)

    def add_q_learning_result(self, model):
        self.q.put(model)

    def save(self, name):
        with open("tuning_results/"+name, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        with open("tuning_results/" + name, "rb") as file:
            results = pickle.load(file)
        return results

    def __getstate__(self):
        return self.__iter__()

    def __setstate__(self, state):
        self.__init__()
        for obj in state:
            self.q.put(obj)


def process_func(params):
    for name, param in zip(TUNING_SETTINGS.keys(), params):
        gs.Q_LEARNING_SETTINGS[name] = param
    q_learning_model = main_app(map_override="10.09;00.53", background=True, end_at_min_epsilon=True, verbose=0)
    print("COMPLETED", params)
    results.add_q_learning_result(q_learning_model)


args = list(it.product(*TUNING_SETTINGS.values()))

results = Results()

if __name__ == '__main__':
    with Pool(12) as pool:
        pool.map(process_func, args)
results.save("lr and target net")
