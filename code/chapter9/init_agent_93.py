import argparse

import h5py

from keras.layers import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD
import dlgo.networks.leaky
from dlgo import agent
from dlgo import encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
    parser.add_argument('output_file')
    args = parser.parse_args()

    # 第九章
    # encoder = encoders.simple.SimpleEncoder((board_size, board_size))
    # model = Sequential()
    # for layer in dlgo.networks.large.layers(encoder.shape()):
    # model.add(layer)
    # model.add(Dense(encoder.num_points()))
    # model.add(Activation('softmax'))
    # new_agent = agent.PolicyAgent(model, encoder)
    # 第十章
    # self._model.compile(
    # loss='categorical_crossentropy',
    # optimizer=SGD(lr=lr, clipnorm=clipnorm))

    encoder = encoders.get_encoder_by_name('simple', args.board_size)
    model = Sequential()
    for layer in dlgo.networks.large.layers(encoder.shape()):
        model.add(layer)
    model.add(Dense(encoder.num_points()))
    model.add(Activation('softmax'))
    #model.compile(loss=agent.policy_gradient_loss, optimizer=opt)
    #model.compile(loss=agent.policy_gradient_loss, optimizer=opt)
    #model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=lr, clipnorm=clipnorm))
    
    
    lr=0.0000001
    clipnorm=1.0
    # 设置默认值
    default_lr = 0.001  # 示例默认学习率
    default_clipnorm = 1.0  # 示例默认梯度裁剪阈值
    # 检查是否传入了自定义值，如果没有，则使用默认值
    lr = lr if lr is not None else default_lr
    clipnorm = clipnorm if clipnorm is not None else default_clipnorm
    opt = SGD(lr=lr, clipnorm=clipnorm)
    model.compile(loss='categorical_crossentropy',optimizer=opt)


    new_agent = agent.PolicyAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()
