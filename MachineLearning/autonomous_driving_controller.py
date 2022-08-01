from App.car_controller import CarControllerKinematic
from MachineLearning.q_learning import CustomModelQLearning, QLearning

import global_settings as gs


class AutonomousDrivingController(CarControllerKinematic):
    def __init__(self, state_n):
        super().__init__()
        self.state = [0 for _ in range(state_n)]

        self.q_learning = CustomModelQLearning(state_n, 5)

        self.brake_amount = 0.1
        self.accelerate_amount = 0.1
        self.steer_amount = 80

        self.max_velocity = 14
        self.max_steering = 80
        self.start_velocity = 7

        self.distance_travelled = 0

        self.current_action = 0

        self.ai_dead = False

        self.current_ep_location_cache = []
        self.previous_ep_location_cache = []

    def end_of_episode(self):
        # New episode so reset controls
        self.steering_angle = 0
        self.velocity = self.start_velocity

        if gs.TRAINING:
            # start training and decay probability
            self.q_learning.decay_exploration_probability()
            self.q_learning.train()

        self.distance_travelled = 0

        self.ai_dead = False

        self.previous_ep_location_cache = self.current_ep_location_cache
        self.current_ep_location_cache = []

    def update_transform(self, velocity_constant):
        """
        actions:
        0: nothing
        1: right steer
        2: left steer
        3: brake
        4: accelerate
        """

        self.distance_travelled += self.velocity*velocity_constant

        action, q_values = self.q_learning.get_action(self.state)

        if action == 0:
            self.steering_angle = 0
            self.current_action = 0

        elif action == 1:
            #self.steering_angle += self.steer_amount
            #self.steering_angle = min(self.max_steering, self.steering_angle)
            self.steering_angle = self.max_steering
            self.current_action = 1

        elif action == 2:
            #self.steering_angle -= self.steer_amount
            #self.steering_angle = max(-self.max_steering, self.steering_angle)
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

        super().update_transform(velocity_constant)

        self.current_ep_location_cache.append(tuple(self.location))

    def end_of_frame(self):
        reward = self.evaluate_reward()

        self.q_learning.update_experience_buffer(self.state, self.current_action, reward)

    def evaluate_reward(self):
        if self.ai_dead:
            return -10

        elif self.current_action == 0:
            return 1
        elif self.current_action == 3:
            return 0.2
        elif self.current_action == 4:
            return 2

        return 1
