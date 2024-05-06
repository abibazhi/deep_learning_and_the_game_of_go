# 到底是fastpolicy呢，还是deeplearningAgent
# 7.3 Training a deep-learning model on human game-play data

# 改了形状和类型
from dlgo.encoders.simple_fastpolicy import SimpleEncoder

# 保存成文件时，变成feature和label，还要改类型！
from verify13_fastpolicy.parallel_processor import GoDataProcessor


from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model
from dlgo.networks import small

from dlgo.agent.pg import PolicyAgent
from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense

import h5py




def train():
    rows, cols = 19, 19
    num_classes = rows * cols
    num_games = 10000


    #encoder = AlphaGoEncoder()
    encoder = SimpleEncoder((rows,cols))
    processor = GoDataProcessor(encoder=encoder.name(),data_directory='data')
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)
    #到这里是生成数据，每次都生成新数据。
    
    # input_shape = (rows, cols, encoder.num_planes)
    # network_layers = small.layers(input_shape)
    # model = Sequential()
    # for layer in network_layers:
    #     model.add(layer)
    # model.add(Dense(num_classes, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

    model = load_model("checkpoints/small_model_epoch_5.keras")

    epochs = 5
    batch_size = 128
    model.fit(
        x=generator.generate(batch_size, num_classes),
        epochs=epochs,
        verbose=1,
        steps_per_epoch=int(generator.get_num_samples() / batch_size),
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=int(test_generator.get_num_samples() / batch_size),
        callbacks=[ModelCheckpoint('./checkpoints/small_model_epoch_{epoch}.keras')]
    )
    

    return
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


train()
