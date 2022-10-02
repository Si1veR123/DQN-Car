from App.car_controller import CarControllerKinematic
from MachineLearning.q_learning import CustomModelQLearning

import global_settings as gs


class AutonomousDrivingController(CarControllerKinematic):
    # base that doesnt include DQN
    def __init__(self, state_n):
        super().__init__()
        self.state = [0 for _ in range(state_n)]

        self.brake_amount = 0.5
        self.accelerate_amount = 0.5
        self.steer_amount = 80

        self.max_velocity = 6
        self.max_steering = 80
        self.start_velocity = 3

        self.distance_travelled = 0

        self.current_action = 0
        self.ai_dead = False

    def end_of_episode(self, verbose=2):
        raise NotImplementedError

    def end_of_frame(self):
        raise NotImplementedError

    def evaluate_reward(self):
        raise NotImplementedError


class AutonomousDrivingControllerCombined(AutonomousDrivingController):
    # driving controller with combined gas and steering networks
    def __init__(self, state_n):
        super().__init__(state_n)
        self.q_learning = CustomModelQLearning(state_n, 3, gs.LOAD_MODEL)

    def end_of_episode(self, verbose=2):
        # New episode so reset controls
        self.steering_angle = 0
        self.velocity = self.start_velocity

        # start training and decay probability
        if gs.Q_LEARNING_SETTINGS["TRAINING"]:
            self.q_learning.decay_exploration_probability()
            self.q_learning.train(verbose)

        self.distance_travelled = 0

        self.ai_dead = False

    def update_transform(self):
        """
        0: nothing
        1: right steer
        2: left steer

        3: brake
        4: accelerate
        """

        self.distance_travelled += self.velocity

        action, _ = self.q_learning.get_action(self.state)

        # Q learning actions
        if action == 0:
            self.steering_angle = 0
            self.current_action = 0

        elif action == 1:
            self.steering_angle = self.max_steering
            self.current_action = 1

        elif action == 2:
            self.steering_angle = -self.max_steering
            self.current_action = 2

        elif action == 3:
            self.velocity -= self.brake_amount

            if self.velocity < 0:
                self.velocity = 0
            self.current_action = 3

        elif action == 4:
            self.velocity += self.accelerate_amount

            if self.velocity > self.max_velocity:
                self.velocity = self.max_velocity
            self.current_action = 4

        self.acceleration = 0
        super().update_transform()

    def end_of_frame(self):
        if gs.Q_LEARNING_SETTINGS["TRAINING"]:
            reward = self.evaluate_reward()
            self.q_learning.update_experience_buffer(self.state, self.current_action, reward)

    def evaluate_reward(self):
        if self.ai_dead:
            return -1

        if self.current_action == 0:
            return 0.01
        elif self.current_action == 1:
            return 0.01
        elif self.current_action == 2:
            return 0.01
        elif self.current_action == 3:
            return 0
        elif self.current_action == 4:
            return 0.02
