from __future__ import print_function
import sys
sys.path.append("/home/jm/goForCheckout/code")
print(sys.path)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.networks.large import layers

go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols
print(111111)
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())

input_channels = encoder.num_planes
input_shape = (input_channels, go_board_rows, go_board_cols)

X, y = processor.load_go_data(num_samples=1)
print("aaaaa")
#X = X.astype('float32')
X = X.astype('int8')
Y = to_categorical(y, nb_classes, dtype="int8")
print("bbbb")

model = Sequential()
network_layers = layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
print("cccc")

model.fit(X, Y, batch_size=128, epochs=100, verbose=1)

weight_file = '../agents/weights.hd5'
model.save_weights(weight_file, overwrite=True)
model_file = '../agents/model.yml'
with open(model_file, 'w') as yml:
    model_yaml = model.to_yaml()
    yml.write(model_yaml)
