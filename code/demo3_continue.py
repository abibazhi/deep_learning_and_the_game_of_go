# 加载最近保存的模型权重
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small,large
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD
import glob
from keras.models import load_model


go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 100
encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name())
generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test', num_games, use_generator=True)
print(1)

latest = sorted(glob.glob('./cks/large_model_epoch_*.h5'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
latest_model_path = latest[-1]  # 获取最后一个checkpoint文件路径

print(latest_model_path)
model = load_model(latest_model_path, compile=False)  # 加载模型权重
print(2)
optimizer = SGD(lr=0.01)
print(3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


print(4)
batch_size = 128
# 继续训练
epochs = 150  # 假设原来总共打算训练50个epoch
last_epoch = int(latest_model_path.split('_')[-1].split('.')[0])  # 获取已训练的最后一个epoch号

remaining_epochs = epochs - last_epoch  # 计算剩余需要训练的epoch数

model.fit_generator(
    generator=generator.generate(batch_size, num_classes),
    epochs=remaining_epochs,
    initial_epoch=last_epoch + 1,  # 设置起始epoch为上次训练终止点的下一个epoch
    steps_per_epoch=generator.get_num_samples() / batch_size,
    validation_data=test_generator.generate(batch_size, num_classes),
    validation_steps=test_generator.get_num_samples() / batch_size,
    callbacks=[ModelCheckpoint('./checkpoints/large_model_epoch_{epoch}.h5')]
)
print(5)

