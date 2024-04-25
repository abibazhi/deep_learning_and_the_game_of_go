import random
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.networks.alphago import alphago_model
from keras.callbacks import ModelCheckpoint 
from dlgo.data.generator import DataGenerator3
from dlgo.data import Sampler 
#from keras import Model
from keras.models import load_model
from keras.optimizers import SGD

import h5py
import sys

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import glob
import numpy as np
import tensorflow as tf


print(sys.path)

class KerasDatasetGenerator:
    def __init__(self, features_path, labels_path, batch_size=128):
        self.features_path = features_path
        self.labels_path = labels_path
        self.batch_size = batch_size

    def load_data(self):
        features = np.load(self.features_path)
        labels = np.load(self.labels_path)
        return features, labels

    def preprocess_data(self, features, labels):
        # 如果需要对数据进行任何预处理（如归一化、标准化等），请在此处添加相应代码
        return features, to_categorical(labels, num_classes=19 * 19)  # 假设您的数据有 19 * 19 个类别

    def create_dataset(self):
        features, labels = self.load_data()
        preprocessed_features, preprocessed_labels = self.preprocess_data(features, labels)

        # 将数据从 channels_first 转换为 channels_last 格式
        preprocessed_features = tf.transpose(preprocessed_features, perm=[0, 2, 3, 1])  # 注意 perm 参数的顺序

        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_features, preprocessed_labels))
        dataset = dataset.shuffle(buffer_size=len(preprocessed_features))  # 随机打乱数据
        dataset = dataset.batch(self.batch_size)  # 分批
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 提前预取数据以提高效率

        return dataset


import glob
import numpy as np
import random
import tensorflow as tf

class KerasMultiFileDatasetGenerator:
    def __init__(self, features_pattern, labels_pattern, batch_size=128, num_files=None):
        self.features_pattern = features_pattern
        self.labels_pattern = labels_pattern
        self.batch_size = batch_size
        self.num_files = num_files
        self.feature_shape = (19, 19, 49)
        self.label_shape = (361,)

    def _load_data_from_single_file(self, file_path):
        return np.load(file_path)

    def _select_files(self, file_list):
        if self.num_files is not None:
            selected_files = random.sample(file_list, k=self.num_files)
            with open('selected_files.txt', 'w') as f:
                for filename in selected_files:
                    f.write(filename + '\n')
            return selected_files
        else:
            return file_list

    def _generate_samples(self, file_list):
        for file_path in file_list:
            data = self._load_data_from_single_file(file_path)
            yield data

    # def preprocess_data(self, features, labels):
    #     # 实现你的预处理逻辑
    #     return preprocessed_features, preprocessed_labels

    def preprocess_data(self, features, labels):
        # 如果需要对数据进行任何预处理（如归一化、标准化等），请在此处添加相应代码
        return features, to_categorical(labels, num_classes=19 * 19)  # 假设您的数据有 19 * 19 个类别


    def create_dataset_generator(self):
        features_files = sorted(glob.glob(self.features_pattern))
        labels_files = sorted(glob.glob(self.labels_pattern))

        selected_features_files = self._select_files(features_files)
        selected_labels_files = self._select_files(labels_files)

        assert len(selected_features_files) == len(selected_labels_files), "Number of selected files must match."

        features_generator = self._generate_samples(selected_features_files)
        labels_generator = self._generate_samples(selected_labels_files)

        for features_batch, labels_batch in zip(features_generator, labels_generator):
            preprocessed_features, preprocessed_labels = self.preprocess_data(features_batch, labels_batch)

            # 将数据从 channels_first 转换为 channels_last 格式
            preprocessed_features = tf.transpose(preprocessed_features, perm=[0, 2, 3, 1])
            print("aaabbbcccc")
            print(preprocessed_features.shape,preprocessed_labels.shape)
            yield preprocessed_features, preprocessed_labels

    def create_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.create_dataset_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.feature_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *self.label_shape), dtype=tf.float32)
            ),
            args=()
        )

        #dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        dataset = dataset.repeat(5)

        return dataset

class KerasMultiFileDatasetGenerator_old(KerasDatasetGenerator):
    def __init__(self, features_pattern, labels_pattern, batch_size=128,num_files=10):
        self.features_pattern = features_pattern
        self.labels_pattern = labels_pattern
        self.batch_size = batch_size
        self.num_files = num_files

    def load_data_from_multiple_files_old(self, pattern):
        file_list = sorted(glob.glob(pattern))  # 获取匹配模式的所有文件，按名称排序
        data_list = [np.load(file) for file in file_list]  # 加载每个文件的内容
        return np.concatenate(data_list, axis=0)  # 沿着第一个轴（样本维度）拼接所有数据

    def load_data_from_multiple_files(self, pattern):
        file_list = sorted(glob.glob(pattern))  # 获取匹配模式的所有文件，按名称排序
        print("aaaaa")
        print(pattern)
        #print(file_list)
        print(self.num_files)


        if self.num_files is not None:
            selected_files = random.sample(file_list, k=self.num_files)
            with open('selected_files.txt', 'w') as f:
                # 遍历数组，将每个文件名写入文件并添加换行符
                for filename in selected_files:
                    f.write(filename + '\n')
        else:
            selected_files = file_list
        data_list = [np.load(file) for file in selected_files]  # 加载选定的文件内容
        return np.concatenate(data_list, axis=0)  # 沿着第一个轴（样本维度）拼接所有数据

    def create_dataset(self):
        #这里执行了两个load,所以文件啊清单是后面那个label的。
        features = self.load_data_from_multiple_files(self.features_pattern)
        labels = self.load_data_from_multiple_files(self.labels_pattern)

        preprocessed_features, preprocessed_labels = self.preprocess_data(features, labels)

        # 将数据从 channels_first 转换为 channels_last 格式
        preprocessed_features = tf.transpose(preprocessed_features, perm=[0, 2, 3, 1])

        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_features, preprocessed_labels))
        dataset = dataset.shuffle(buffer_size=len(preprocessed_features))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        dataset = dataset.repeat(100)

        return dataset


def train():
    rows, cols = 19, 19
    num_classes = rows * cols
    num_games = 10000


    encoder = AlphaGoEncoder()
    #processor = GoDataProcessor(encoder=encoder.name())
    #generator = processor.load_go_data('train', num_games, use_generator=True)
    #test_generator = processor.load_go_data('test', num_games, use_generator=True)
    print("生成数据就好了！")
    print("下面就光是训练")
    print(encoder.num_planes)

    #input_shape = ( rows, cols, encoder.num_planes)
    #alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)
    #alphago_sl_policy.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])

    #latest = sorted(glob.glob('../checkpoints/alphago*.keras'), key=lambda x: int(x.split('.')[0]))
    #latest_model_path = latest[-1]  # 获取最后一个checkpoint文件路径

    latest_model_path = "../checkpoints/alphago_gen_90.keras"
    print(latest_model_path)
    alphago_sl_policy = load_model(latest_model_path, compile=False)  # 加载模型权重
    print(2)
    optimizer = SGD(learning_rate=0.05)
    print(3)
    alphago_sl_policy.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



    epochs = 5
    batch_size = 256


    features_pattern = "./code/data/*train_features*.npy"
    labels_pattern = "./code/data/*train_labels*.npy"
    # 抽取10个文件作为数据集
    generator = KerasMultiFileDatasetGenerator(features_pattern, labels_pattern, batch_size=128, num_files=25)
    train_dataset = generator.create_dataset()
    #dataset = generator.create_dataset()

    total_features_count = 0
    for features, labels in train_dataset:
        total_features_count += features.shape[0]
    print(f"Total number of features in the dataset: {total_features_count}")

    steps_per_epoch = int(total_features_count/batch_size/epochs)
    steps_per_epoch = 100
    print(batch_size, steps_per_epoch)


    # 使用 `dataset` 对象进行训练，例如：
    for epoch in range(epochs):
        for features_batch, labels_batch in train_dataset:
            alphago_sl_policy.train_on_batch(features_batch, labels_batch)

    # alphago_sl_policy.fit(train_dataset, 
    #                       epochs=epochs, 
    #                       steps_per_epoch=steps_per_epoch, 
    #                       verbose=1,
    #                       callbacks=[ModelCheckpoint('../checkpoints/alphago_{epoch}.keras')]
    #                       )    

    # alphago_sl_policy.fit(
    #     generator=generator.generate(batch_size, num_classes),
    #     epochs=epochs,
    #     steps_per_epoch=generator.get_num_samples() / batch_size,
    #     validation_data=test_generator.generate(batch_size, num_classes),
    #     validation_steps=test_generator.get_num_samples() / batch_size,
    #     callbacks=[ModelCheckpoint('alphago_sl_policy_{epoch}.keras')]
    # )
    # alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)
    # with h5py.File('alphago_sl_policy.h5', 'w') as sl_agent_out:
    #     alphago_sl_agent.serialize(sl_agent_out)
    #alphago_sl_agent.serialize('alphago_sl_policy.h5')


train()
#predict()
