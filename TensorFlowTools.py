import tensorflow as tf
import re
import os


class ModelCheckPointSaver(tf.keras.callbacks.Callback):
    def __init__(self, model_id, checkpoint_dir, max_num_weights=5, save_weights=True):
        super().__init__()

        self.checkpoint_dir = checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.max_num_weights = max_num_weights
        self.model_id = model_id
        self.save_weights = save_weights

    def on_epoch_end(self, epoch, logs=None):
        min_acc_file, max_acc_file, max_acc, num_weights = self.scan_weight_files()
        cur_acc = logs["val_sparse_categorical_accuracy"]
        if cur_acc > max_acc:
            if self.save_weights:
                save_name = "weights.e{}-acc{}.h5".format(
                    epoch, cur_acc
                )
                self.model.save_weights(
                    self.checkpoint_dir + os.sep + self.model_id + "." + save_name
                )
            else:
                save_name = "model.e{}-acc{}.h5".format(
                    epoch, cur_acc
                )
                self.model.save(
                    self.checkpoint_dir + os.sep + self.model_id + "." + save_name
                )
            if self.max_num_weights and num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)

    def scan_weight_files(self):
        min_acc = float('inf')
        max_acc = 0.0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        files = os.listdir(self.checkpoint_dir)

        for file in files:
            if file.endswith(".h5"):
                num_weights += 1
                file_acc = float(
                    re.findall(
                        r"[+-]?\d+\.\d+", file.split("acc")[-1]
                    )[0]
                )
                if file_acc > max_acc:
                    max_acc = file_acc
                    max_acc_file = file
                if file_acc < min_acc:
                    min_acc = file_acc
                    min_acc_file = file

            return min_acc_file, max_acc_file, max_acc, num_weights
