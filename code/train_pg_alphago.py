from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model
from keras.callbacks import ModelCheckpoint 
import h5py



rows, cols = 19, 19
num_classes = rows * cols
num_games = 10000
encoder = AlphaGoEncoder()
processor = GoDataProcessor(encoder=encoder.name())
generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test', num_games, use_generator=True)


input_shape = (encoder.num_planes, rows, cols)
alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)
alphago_sl_policy.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

epochs = 200
batch_size = 128
alphago_sl_policy.fit_generator(
    generator=generator.generate(batch_size, num_classes),
    epochs=epochs,
    steps_per_epoch=generator.get_num_samples() / batch_size,
    validation_data=test_generator.generate(batch_size, num_classes),
    validation_steps=test_generator.get_num_samples() / batch_size,
    callbacks=[ModelCheckpoint('alphago_sl_policy_{epoch}.h5')]
)
alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)
with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
    alphago_sl_agent.serialize(sl_agent_out)


# import argparse

# import h5py

# from dlgo import agent
# from dlgo import rl


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--learning-agent', required=True)
#     parser.add_argument('--agent-out', required=True)
#     parser.add_argument('--lr', type=float, default=0.0001)
#     parser.add_argument('--bs', type=int, default=512)
#     parser.add_argument('experience', nargs='+')

#     args = parser.parse_args()

#     learning_agent = agent.load_policy_agent(h5py.File(args.learning_agent))
#     for exp_filename in args.experience:
#         print('Training with %s...' % exp_filename)
#         exp_buffer = rl.load_experience(h5py.File(exp_filename))
#         learning_agent.train(exp_buffer, lr=args.lr, batch_size=args.bs)

#     with h5py.File(args.agent_out, 'w') as updated_agent_outf:
#         learning_agent.serialize(updated_agent_outf)


# if __name__ == '__main__':
#     main()
