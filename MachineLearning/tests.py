from q_learning import CustomModelQLearning
from neural_network_classes import NeuralNetwork, ConnectedLayer, GradientDescent, relu


def test_target_net():
    # tests that target net works correctly
    q_learning = CustomModelQLearning(5, 3, model_type="combined")

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


def test_learning_rate():
    net = NeuralNetwork(
        [
            ConnectedLayer(relu, 3, 6),
            ConnectedLayer(relu, 6, 18),
            ConnectedLayer(relu, 18, 6),
            ConnectedLayer(relu, 6, 2)
        ],
        learning_rate=0
    )

    predicts_1 = net.predict([5, 15, 25])

    net.train([[1, 2, 3], [4, 5, 6]], [[10, 11], [13, 14]], epochs=1, log=False)

    predicts_2 = net.predict([5, 15, 25])

    print(predicts_1, predicts_2)  # with learning rate 0, should be same


test_learning_rate()
test_target_net()
