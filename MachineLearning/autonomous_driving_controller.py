from App.car_controller import CarControllerKinematic
from MachineLearning.q_learning import CustomModelQLearning, QLearning

import global_settings as gs


class AutonomousDrivingController(CarControllerKinematic):
    def __init__(self, state_n):
        super().__init__()
        self.state = [0 for _ in range(state_n)]

        self.steer_q_learning = CustomModelQLearning(state_n, 3, gs.LOAD_MODEL_STEER, "STEER")
        self.gas_q_learning = CustomModelQLearning(state_n, 3, gs.LOAD_MODEL_GAS, "GAS")

        self.brake_amount = 0.1
        self.accelerate_amount = 0.1
        self.steer_amount = 80

        self.max_velocity = 6
        self.max_steering = 80
        self.start_velocity = 1.5

        self.distance_travelled = 0

        self.current_action = [0, 0]  # [steer action, gas action]

        self.ai_dead = False

    def end_of_episode(self):
        # New episode so reset controls
        self.steering_angle = 0
        self.velocity = self.start_velocity

        # start training and decay probability
        if gs.get_q_learning_settings("STEER")["TRAINING"]:
            self.steer_q_learning.decay_exploration_probability()
            print("STEERING TRAINING")
            self.steer_q_learning.train()

        if gs.get_q_learning_settings("GAS")["TRAINING"]:
            self.gas_q_learning.decay_exploration_probability()
            print("GAS TRAINING")
            self.gas_q_learning.train()

        self.distance_travelled = 0

        self.ai_dead = False

    def update_transform(self, velocity_constant):
        """
        steer_actions:
        0: nothing
        1: right steer
        2: left steer

        gas_actions:
        0: nothing
        1: brake
        2: accelerate
        """

        self.distance_travelled += self.velocity*velocity_constant

        gas_action, _ = self.gas_q_learning.get_action(self.state)
        steer_action, _ = self.steer_q_learning.get_action(self.state)

        # steer Q learning actions
        if steer_action == 0:
            self.steering_angle = 0
            self.current_action[0] = 0

        elif steer_action == 1:
            self.steering_angle = self.max_steering
            self.current_action[0] = 1

        elif steer_action == 2:
            self.steering_angle = -self.max_steering
            self.current_action[0] = 2

        # gas Q learning actions
        if gas_action == 0:
            self.current_action[1] = 0

        elif gas_action == 1:
            # self.velocity -= self.brake_amount

            if self.velocity < 0:
                self.velocity = 0
            self.current_action[1] = 1

        elif gas_action == 2:
            # self.velocity += self.accelerate_amount

            if self.velocity > self.max_velocity:
                self.velocity = self.max_velocity
            self.current_action[1] = 2

        self.acceleration = 0
        super().update_transform(velocity_constant)

    def end_of_frame(self):
        g_reward, s_reward = self.evaluate_reward(True), self.evaluate_reward(False)

        if gs.get_q_learning_settings("GAS")["TRAINING"]:
            self.gas_q_learning.update_experience_buffer(self.state, self.current_action[1], g_reward)

        if gs.get_q_learning_settings("STEER")["TRAINING"]:
            self.steer_q_learning.update_experience_buffer(self.state, self.current_action[0], s_reward)

    def evaluate_reward(self, gas):
        if self.ai_dead:
            return -10
        if gas:
            if self.current_action[1] == 0:
                # nothing
                return 1
            if self.current_action[1] == 1:
                # brake
                return 0.8
            if self.current_action[1] == 2:
                # accelerate
                return 1.2
        else:
            # steering left or right
            return 1
