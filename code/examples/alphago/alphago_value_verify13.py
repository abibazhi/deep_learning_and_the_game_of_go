# 这个文件中的value，就是指Q吧。
# tag::init_value[]
from dlgo.networks.alphago import alphago_model
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.rl import ValueAgent, load_experience
import h5py

rows, cols = 19, 19
encoder = AlphaGoEncoder()
#input_shape = (encoder.num_planes, rows, cols)
input_shape = (rows, cols, encoder.num_planes)
alphago_value_network = alphago_model(input_shape)

alphago_value = ValueAgent(alphago_value_network, encoder)
# end::init_value[]

# tag::train_value[]
experience = load_experience(h5py.File('verify13_improve/alphago_rl_experience.h5', 'r'))

alphago_value.train(experience)

with h5py.File('alphago_value.h5', 'w') as value_agent_out:
    alphago_value.serialize(value_agent_out)
# end::train_value[]
