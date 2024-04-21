import numpy as np
import tensorflow as tf

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

        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_features, preprocessed_labels))
        dataset = dataset.shuffle(buffer_size=len(preprocessed_features))  # 随机打乱数据
        dataset = dataset.batch(self.batch_size)  # 分批
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # 提前预取数据以提高效率

        return dataset

def main():
    features_path = "../data/KGS-2007-19-11644-train_features_100.npy"
    labels_path = "../data/KGS-2007-19-11644-train_labels_100.npy"

    generator = KerasDatasetGenerator(features_path, labels_path, batch_size=128)
    train_dataset = generator.create_dataset()

    # 现在您可以使用 train_dataset 作为 Keras 模型的输入数据集进行训练
    # model.fit(train_dataset, epochs=.
main()
