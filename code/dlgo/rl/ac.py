import numpy as np

from keras.optimizers import SGD

from .. import encoders
from .. import goboard
from .. import kerasutil
from ..agent import Agent
from ..agent.helpers import is_point_an_eye

__all__ = [
    'ACAgent',
    'load_ac_agent',
]


class ACAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 1 # 这个没有用，人家训练时设置的。
        # temperature从1改为0.9，减少随机性。希望loss稳定。
        # temperature从0.9改为0.8，继续减少随机性。希望loss更加稳定。
        # 还是改回0.9吧。其实还是不清楚。

        self.last_state_value = 0

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height

        board_tensor = self.encoder.encode(game_state)
        x = np.array([board_tensor])

        actions, values = self.model.predict(x)
        move_probs = actions[0]
        estimated_value = values[0][0]
        self.last_state_value = float(estimated_value)

        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = np.power(move_probs, 1.0 / self.temperature)
        move_probs = move_probs / np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board,
                                        point,
                                        game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(
                        state=board_tensor,
                        action=point_idx,
                        estimated_value=estimated_value
                    )
                return goboard.Move.play(point)
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()

    def train(self, experience, lr=0.1, batch_size=128):
        opt = SGD(lr=lr, clipvalue=0.2)
        self.model.compile(
            optimizer=opt,
            loss=['categorical_crossentropy', 'mse'],loss_weights=[1, 0.25])

        n = experience.states.shape[0]
        batch_size = int(n/120*10) + 1
        print(f"batchsize={batch_size},lr={lr}")


        num_moves = self.encoder.num_points()
        policy_target = np.zeros((n, num_moves))
        value_target = np.zeros((n,))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            policy_target[i][action] = experience.advantages[i]
            value_target[i] = reward

        history = self.model.fit(
            experience.states,
            [policy_target, value_target],
            batch_size=batch_size,
            epochs=1)

        with open("chapter12/log.txt", 'a') as logf:
            logf.write('---loss-------------------\n')
            for epoch, loss in enumerate(history.history['loss']):
                log_entry = f"Epoch {epoch + 1}: Loss = {loss}\n"
                logf.write(log_entry)
                print(log_entry)
            
            for key, value in history.history.items():
                print(f"{key}: {value}")
                logf.write(f"{key}: {value}")


    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

    def diagnostics(self):
        return {'value': self.last_state_value}


def load_ac_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = encoders.get_encoder_by_name(
        encoder_name,
        (board_width, board_height))
    return ACAgent(model, encoder)
