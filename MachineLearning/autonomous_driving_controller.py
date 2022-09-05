from App.car_controller import CarControllerKinematic
from MachineLearning.q_learning import CustomModelQLearning

import global_settings as gs


class AutonomousDrivingController(CarControllerKinematic):
    # base that doesnt include DQN
    def __init__(self, state_n):
        super().__init__()
        self.state = [0 for _ in range(state_n)]

        self.brake_amount = 0.03
        self.accelerate_amount = 0.03
        self.steer_amount = 80

        self.max_velocity = 6
        self.max_steering = 80
        self.start_velocity = 1.5

        self.distance_travelled = 0

        self.current_action = 0
        self.ai_dead = False

    def end_of_episode(self):
        raise NotImplementedError

    def end_of_frame(self):
        raise NotImplementedError

    def evaluate_reward(self, gas):
        raise NotImplementedError


class AutonomousDrivingControllerSeparate(AutonomousDrivingController):
    # driving controller with separate gas and steering networks
    def __init__(self, state_n):
        super().__init__(state_n)

        self.steer_q_learning = CustomModelQLearning(state_n, 3, gs.LOAD_MODEL_STEER, "STEER")
        self.gas_q_learning = CustomModelQLearning(state_n, 3, gs.LOAD_MODEL_GAS, "GAS")

        self.current_action = [0, 0]  # [steer action, gas action]

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
            self.velocity -= self.brake_amount

            if self.velocity < 0:
                self.velocity = 0
            self.current_action[1] = 1

        elif gas_action == 2:
            self.velocity += self.accelerate_amount

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
        if gas:
            if self.ai_dead:
                return -10
            if self.current_action[1] == 0:
                # nothing
                return 1
            if self.current_action[1] == 1:
                # brake
                return 0.7
            if self.current_action[1] == 2:
                # accelerate
                return 1.3
        else:
            # steering left or right
            if self.ai_dead:
                return -10

            return 1


class AutonomousDrivingControllerCombined(AutonomousDrivingController):
    # driving controller with combined gas and steering networks
    def __init__(self, state_n):
        super().__init__(state_n)
        self.q_learning = CustomModelQLearning(state_n, 5, gs.LOAD_MODEL_COMBINED, "COMBINED")

    def end_of_episode(self):
        # New episode so reset controls
        self.steering_angle = 0
        self.velocity = self.start_velocity

        # start training and decay probability
        if gs.get_q_learning_settings("COMBINED")["TRAINING"]:
            self.q_learning.decay_exploration_probability()
            print("COMBINED TRAINING")
            self.q_learning.train()

        self.distance_travelled = 0

        self.ai_dead = False

    def update_transform(self, velocity_constant):
        """
        0: nothing
        1: right steer
        2: left steer

        3: brake
        4: accelerate
        """

        self.distance_travelled += self.velocity*velocity_constant

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
        super().update_transform(velocity_constant)

    def end_of_frame(self):
        reward = self.evaluate_reward()

        if gs.get_q_learning_settings("COMBINED")["TRAINING"]:
            self.q_learning.update_experience_buffer(self.state, self.current_action, reward)

    def evaluate_reward(self):
        if self.ai_dead:
            return -10

        if self.current_action == 0:
            return 1
        elif self.current_action == 1:
            return 1
        elif self.current_action == 2:
            return 1
        elif self.current_action == 3:
            return 0.7
        elif self.current_action == 4:
            return 1.3
