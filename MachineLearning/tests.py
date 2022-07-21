from MachineLearning.q_learning import CustomModelQLearning


def test_target_net():
    q_learning = CustomModelQLearning(5, 3)

    normal_q = q_learning.get_q_values((1, 1, 1, 1, 1), target=False)
    target_q = q_learning.get_q_values((1, 1, 1, 1, 1), target=True)

    assert normal_q == target_q

    q_learning.experience_buffer = [
        ((10, 15, 20, 25, 30), 2, 5),
        ((10, 15, 20, 25, 30), 1, 5),
        ((10, 15, 20, 25, 30), 1, 2),
        ((10, 15, 20, 25, 30), 0, -10)
    ]
    q_learning.train()

    normal_q_trained = q_learning.get_q_values((1, 1, 1, 1, 1), target=False)
    target_q_trained = q_learning.get_q_values((1, 1, 1, 1, 1), target=True)

    assert normal_q_trained != target_q_trained

    q_learning.update_target_network()

    normal_q_updated = q_learning.get_q_values((1, 1, 1, 1, 1), target=False)
    target_q_updated = q_learning.get_q_values((1, 1, 1, 1, 1), target=True)

    assert normal_q_updated == target_q_updated


test_target_net()
