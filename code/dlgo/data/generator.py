import tensorflow as tf
import glob
import numpy as np
from keras.utils import to_categorical
import numpy as np
from keras.utils import to_categorical

class DataGenerator3:
    def __init__(self, data_directory, samples, max_files=1):
        self.data_directory = data_directory
        self.samples = samples
        self.max_files = max_files
        self.files = list(zip_file_name for zip_file_name, index in samples)[:max_files]

    def build_dataset(self, batch_size=128, num_classes=19 * 19):


        def load_and_assemble_data(feature_files, max_files, batch_size):
            data_list = []
            labels_list = []
            num_files_loaded = 0

            for file_path in feature_files:
                if num_files_loaded >= max_files:
                    break

                x, y = _load_data(file_path)
                data_list.append(x)
                labels_list.append(y)
                num_files_loaded += 1

            # Concatenate loaded data into single arrays
            data = np.concatenate(data_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)

            # Optionally shuffle the data if desired
            indices = np.arange(len(data))
            np.random.shuffle(indices)

            data = data[indices]
            labels = labels[indices]

            # Split into batches
            num_batches = len(data) // batch_size
            data_batches = np.split(data[:num_batches * batch_size], num_batches)
            labels_batches = np.split(labels[:num_batches * batch_size], num_batches)

            return list(zip(data_batches, labels_batches))


        # # Assuming you have a list of feature files (feature_files) and a defined model
        # # Set your batch size and max number of files to load here
        # batch_size = 32
        # max_files_to_load = 1  # Start with loading 1 file, increase as needed

        # data_batches, labels_batches = load_and_assemble_data(feature_files, max_files_to_load, batch_size)

        # Train the model
        #history = model.fit(x=data_batches, y=labels_batches, batch_size=batch_size, epochs=num_epochs, ...)


        def _load_data(file_path1):
            print(file_path1)
            x = np.load(file_path1)
            label_file = file_path1.replace('features', 'labels')
            y = np.load(label_file)
            return x.astype('float32'), to_categorical(y.astype(int), num_classes)

        def load_and_preprocess(zip_file_name):
            base = self.data_directory + '/' + zip_file_name.replace('.tar.gz', '') + 'train_features_*.npy'
            feature_files = glob.glob(base)

            if len(feature_files) == 0:
                print("特征文件未生成")
                return None

            print("特征文件打印：")
            print(feature_files)

            # 调用 load_and_assemble_data 函数处理数据集
            data_batches, labels_batches = load_and_assemble_data(feature_files, batch_size, num_classes)
            return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(data_batches),
                                        tf.data.Dataset.from_tensor_slices(labels_batches)))

        datasets = []
        for zip_file_name in self.files:
            dataset = load_and_preprocess(zip_file_name)
            if dataset is not None:
                datasets.append(dataset)

        return tf.data.Dataset.zip(tuple(datasets))

    def generate(self, batch_size=128, num_classes=19 * 19):
        return self.build_dataset(batch_size, num_classes)
    

class DataGenerator1:
    def __init__(self, data_directory, samples):
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)

    def build_dataset(self, batch_size=128, num_classes=19 * 19):
        def _load_and_preprocess(file_name):
            print('bbbbbbb')
            print(file_name)
            file_name = file_name.replace('.tar.gz', '') + 'train'
            print(file_name)
            base = self.data_directory + '/' + file_name + '_features_*.npy'
            print(base)

            def _load_data(file_path1):
                print('aaaaaaa')
                print(file_path1)
                x = np.load(file_path1)
                print(f"这个是从文件加载出来的{x.shape}")
                label_file = file_path1.replace('features', 'labels')
                y = np.load(label_file)
                return x.astype('float32'), to_categorical(y.astype(int), num_classes)

            feature_files = glob.glob(base)
            if len(feature_files) == 0:
                print("特征文件未生成")
                return
            else:
                print("特征文件打印：")
                print(feature_files)
            
            dataset = tf.data.Dataset.from_tensor_slices(feature_files)
            # 使用 lambda 表达式显式传递文件路径给 _load_data 函数
            dataset = dataset.map(lambda file_path1: _load_data(file_path1), num_parallel_calls=tf.data.AUTOTUNE)
            #dataset = tf.data.Dataset.from_tensor_slices(feature_files)
            #dataset = dataset.map(_load_data, num_parallel_calls=tf.data.AUTOTUNE)

            # # Ensure each batch contains at least one sample
            # min_after_dequeue = 1000
            # capacity = min_after_dequeue + 3 * batch_size
            # dataset = dataset.shuffle(min_after_dequeue, seed=None).batch(batch_size).repeat()

            return dataset

        datasets = []
        for zip_file_name in self.files:
            dataset = _load_and_preprocess(zip_file_name)
            if dataset != None:
                datasets.append(dataset)

        return tf.data.Dataset.zip(tuple(datasets))

    def generate(self, batch_size=128, num_classes=19 * 19):
        return self.build_dataset(batch_size, num_classes)

class DataGenerator:
    def __init__(self, data_directory, samples):
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)  # <1>
        self.num_samples = None

    def get_num_samples(self, batch_size=128, num_classes=19 * 19):  # <2>
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size, num_classes=num_classes):
                self.num_samples += X.shape[0]
            return self.num_samples
# <1> Our generator has access to a set of files that we sampled earlier.
# <2> Depending on the application, we may need to know how many examples we have.
# end::data_generator[]

# tag::private_generate[]
    def _generate(self, batch_size, num_classes):
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = self.data_directory + '/' + file_name + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                print(f"这个是从文件加载出来的{x.shape}")
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), num_classes)
                while x.shape[0] >= batch_size:
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    #print(x_batch.shape) # (128,19,19,1)
                    #print(y_batch.shape) # (128,361)
                    yield x_batch, y_batch  # <1>

# <1> We return or "yield" batches of data as we go.
# end::private_generate[]

# tag::generate[]
    def generate(self, batch_size=128, num_classes=19 * 19):
        while True:
            for item in self._generate(batch_size, num_classes):
                yield item
# end::generate[]
