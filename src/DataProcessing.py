import re
import tensorflow as tf
import os
import random
from skimage import io
from collections import Counter, defaultdict
from skimage.transform import resize
from joblib import Parallel, delayed
from ProcessARGS import TFRecordConfigArgs, GlobalArgs


def data_parser(x):
    """
    :param x:
    :return:
    """
    description = {
        'height': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'width': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'channel': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        'img_raw': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    parsed_features = tf.io.parse_single_example(x, description)
    height, width, channel, label = [
        parsed_features[x] for x in ['height', 'width', 'channel', 'label']
    ]
    img = tf.io.decode_raw(parsed_features['img_raw'], 'float32')
    img = tf.reshape(img, (*TFRecordConfigArgs.image_size, TFRecordConfigArgs.num_channels))
    img.set_shape((*TFRecordConfigArgs.image_size, TFRecordConfigArgs.num_channels))
    return (
        img, label
    )


def get_data(file_path, buffer_size, batch_size, auto_tune):
    return tf.data.TFRecordDataset(
        file_path,
        num_parallel_reads=auto_tune
    ).map(
        data_parser,
        num_parallel_calls=auto_tune
    ).repeat().shuffle(
        buffer_size=buffer_size
    ).batch(
        batch_size=batch_size
    ).prefetch(
        auto_tune
    )


def train_validation_split(file_names, split_rate, file_type='tfrecords'):
    random.Random(GlobalArgs.random_seed).shuffle(file_names)
    if file_type:
        file_names = [x for x in file_names if x.endswith(file_type)]
    split_index = int(len(file_names) * split_rate)
    return (
        file_names[split_index:],
        file_names[:split_index]
    )


class TFRecordData:
    def __init__(
            self,
            directory,
            save_directory,
            size,
            image_size
    ):
        """
        :param directory:
        :param save_directory:
        :param size:
        :param image_size:
        """
        self.directory = directory
        self.save_directory = save_directory
        self.image_size = image_size
        self.size = size
        print('Analyzing image files and constructing distribution...')
        self.__process_categories()

    def convert(self):
        for i in range(self.num_iter):
            print('Writing {} of {}...'.format(i + 1, self.num_iter))
            sub_files = sum(
                [
                    self.category_to_file[category][i * self.num_per_category: (i + 1) * self.num_per_category]
                    for category in self.category_to_file
                ],
                []
            )

            # random.Random(ARGS.GlobalArgs['random_seed']).shuffle(files)  # reproduce

            print('Loading Files...')
            train_x = Parallel(n_jobs=-1)(
                delayed(self.helper)(f)
                for f in sub_files
            )
            train_y = [self.__extract_class(f) for f in sub_files]

            print('Initializing TFRecord Data Writer...')
            cur_output_name = self.save_directory + "/" + '{}_{}'.format(i, i + 1) + '.tfrecords'
            writer = tf.io.TFRecordWriter(cur_output_name)

            for x, y in zip(train_x, train_y):
                if x is None:
                    continue
                n_row, n_col, depth = x.shape
                img_raw = x.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': self.__int64_feature(n_row),
                            'width': self.__int64_feature(n_col),
                            'channel': self.__int64_feature(depth),
                            'label': self.__int64_feature(y),
                            'img_raw': self.__bytes_feature(img_raw)
                        }
                    )
                )
                writer.write(example.SerializeToString())
            writer.close()

    def helper(self, x):
        try:
            img = resize(
                io.imread(x),
                self.image_size,
                anti_aliasing=True
            ).astype('float32')
        except:
            img = None
        return img

    def __process_categories(self):
        original_files = os.listdir(self.directory)
        self.category_to_file = defaultdict(list)
        self.files = ["{}/{}".format(self.directory, x) for x in original_files]
        self.num_files = len(self.files)
        self.category = [self.__extract_class(x) for x in original_files]
        self.category_count = Counter(self.category)
        self.num_category = len(self.category_count)
        for file, category in zip(self.files, self.category):
            self.category_to_file[category].append(file)
        for category in self.category_to_file:
            random.Random(
                GlobalArgs.random_seed
            ).shuffle(
                self.category_to_file[category]
            )
        self.min_class = min(self.category_count.values())
        self.num_per_category = self.size // self.num_category
        self.num_iter = self.min_class * self.num_category // self.size + 1
        print("Found {} categories: {}...".format(self.num_category, self.category_count))

    @staticmethod
    def __extract_class(x):
        # class 2 and 3 are considered as same
        # 2, 3 -> 1; 1 -> 0; 4 -> 2
        return int(x.split('/')[-1].split('_')[1]) // 2

    @staticmethod
    def __int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def __bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def __float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def get_optimal_model(model_dir):
    model_files = os.listdir(model_dir)
    if not model_files:
        raise FileNotFoundError("File no found.")
    else:
        model_accs = [
            float(
                re.findall(
                    r"[+-]?\d+\.\d+", file.split("acc")[-1]
                )[0]
            )
            for file in model_files
        ]
        optimal_model_dir = model_files[model_accs.index(max(model_accs))]
    return model_dir + os.sep + optimal_model_dir
