import random
from itertools import product
from ssbm_gym.ssbm import SimpleButton, SimpleController, RealControllerState


class ActionSpace():
    def __init__(self):
        self.controller = self.make_controller()
        self.actions = [a.real_controller for a in self.controller]
        self.n = len(self.controller)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def sample(self):
        return random.randrange(self.n)

    def neutral(self):
        return 0

    def from_index(self, n):
        return self.actions[n]

    def make_controller(self):
        controller = []
        for button, stick in enumerate([NONE_stick, A_stick, B_stick, Z_stick, Y_stick, L_stick]):
            controller += [SimpleController.init(*args) for args in product([SimpleButton(button)], stick)]
        return controller


class SimpleActionSpace():
    def __init__(self):
        self.controller = self.make_controller()
        self.actions = [a.real_controller for a in self.controller]
        self.n = len(self.controller)

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def sample(self):
        return random.randrange(self.n)

    def from_index(self, n):
        return self.actions[n]

    def make_controller(self):
        controller = []
        for button, stick in zip([0, 1, 4], [NONE_stick, A_stick, Y_stick]):
            controller += [SimpleController.init(*args) for args in product([SimpleButton(button)], stick)]
        return controller


NONE_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.0, 0.5),
    (.35, 0.5),
    (.65, 0.5),
    (1.0, 0.5)
]
A_stick = [
    (0.5, 0.0),
    (0.0, 0.5),
    (.35, 0.5),
    (0.5, 0.5),
    (.65, 0.5),
    (1.0, 0.5),
    (0.5, .35),
    (0.5, .65),
    (0.5, 1.0)
]
B_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (0.0, 0.5),
    (1.0, 0.5)
]
Z_stick = [
    (0.5, 0.5)
]
Y_stick = [
    (0.0, 0.5),
    (0.5, 0.5),
    (1.0, 0.5)
]
L_stick = [
    (0.5, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (.075, 0.25),
    (.925, 0.25)
]
