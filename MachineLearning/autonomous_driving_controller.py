from App.car_controller import CarControllerKinematic
from MachineLearning.q_learning import CustomModelQLearning, QLearning

import global_settings as gs


class AutonomousDrivingController(CarControllerKinematic):
    def __init__(self, state_n):
        super().__init__()
        self.state = [0 for _ in range(state_n)]

        self.q_learning = CustomModelQLearning(state_n, 3)

        self.brake_amount = 3
        self.accelerate_amount = 5
        self.steer_amount = 80

        self.max_velocity = 200
        self.max_steering = 80
        self.start_velocity = 10

        self.distance_travelled = 0

        self.current_action = 0

        self.ai_dead = False

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

    def update_transform(self, velocity_constant):
        """
        actions:
        0: nothing
        1: brake
        2: accelerate
        3: steer left
        4: steer right
        """

        self.distance_travelled += self.velocity*velocity_constant

        action, q_values = self.q_learning.get_action(self.state)

        if action == 10:
            self.velocity -= self.brake_amount

            if self.velocity < 0:
                self.velocity = 0

        elif action == 10:
            self.velocity += self.accelerate_amount

            if self.velocity > self.max_velocity:
                self.velocity = self.max_velocity

        elif action == 0:
            self.steering_angle = 0
            self.current_action = 0

        elif action == 1:
            #self.steering_angle += self.steer_amount
            #self.steering_angle = min(self.max_steering, self.steering_angle)
            self.steering_angle = self.max_steering
            self.current_action = 3

        elif action == 2:
            #self.steering_angle -= self.steer_amount
            #self.steering_angle = max(-self.max_steering, self.steering_angle)
            self.steering_angle = -self.max_steering
            self.current_action = 4

        super().update_transform(velocity_constant)

        reward = self.evaluate_reward()

        self.q_learning.update_experience_buffer(self.state, action, reward)

    def evaluate_reward(self):
        if self.ai_dead:
            return -10

        return 1
