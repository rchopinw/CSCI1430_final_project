import os
from DataProcessing import train_validation_split, get_data, get_optimal_model
import tensorflow as tf
from Models import ModelCollection
from ARGS import ARGS
import argparse


def model_parser():
    parser = argparse.ArgumentParser(
        description="Face Mask Detection Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mode',
        required=True,
        choices=["train", "test"],
        help="Please specify task from [train, test]."
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=['Vanilla', "ResNet50", "Xception", "VGG19", "InceptionV3", "InceptionResNetV2", "All"],
        help="Please select model from [Vanilla, ResNet50, Xception, VGG19, InceptionV3, InceptionResNetV2, All] to train."
    )
    parser.add_argument(
        '--testFile',
        required=False,
        default=ARGS.GlobalArgs["test_save_directory"],
        help="Please specify a file path / directory to evaluate, accepted file types: [tfrecords]."
    )
    return parser.parse_args()


def train_model(model_name, model):
    print("...Training {} Model...".format(model_name))
    tf_rec_files = [
        ARGS.TFRecordConfig["save_directory"] + os.sep + x
        for x in os.listdir(ARGS.TFRecordConfig["save_directory"])
    ]
    train_files, validation_files = train_validation_split(
        file_names=tf_rec_files,
        split_rate=ARGS.GlobalArgs["validation_split"],
        file_type='tfrecords'
    )
    train_data = get_data(
        file_path=train_files,
        buffer_size=ARGS.__dict__[model_name]["train_buffer_size"],
        batch_size=ARGS.__dict__[model_name]["train_batch_size"],
        auto_tune=ARGS.__dict__[model_name]["auto_tune"]
    )
    validation_data = get_data(
        file_path=validation_files,
        buffer_size=ARGS.__dict__[model_name]["validation_buffer_size"],
        batch_size=ARGS.__dict__[model_name]["validation_batch_size"],
        auto_tune=ARGS.__dict__[model_name]["auto_tune"]
    )

    model.compile(
        optimizer=ARGS.__dict__[model_name]["train_optimizer"],
        loss=ARGS.__dict__[model_name]["train_loss"],
        metrics=ARGS.__dict__[model_name]["metrics"]
    )

    steps_per_epoch = len(train_files) * ARGS.TFRecordConfig["size"] // ARGS.__dict__[model_name]["train_batch_size"] + 1
    validation_steps = \
        len(validation_files) * ARGS.TFRecordConfig["size"] // ARGS.__dict__[model_name]["validation_batch_size"] + 1

    model.fit(
        x=train_data,
        validation_data=validation_data,
        epochs=ARGS.__dict__[model_name]["train_epoch"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=ARGS.__dict__[model_name]["callbacks"]
    )


def train_vanilla():
    vanilla_model = mc.get_vanilla_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs["num_classes"],
        resize=ARGS.VanillaModel["resize"],
        translation=ARGS.VanillaModel["translation"],
        zoom=ARGS.VanillaModel["zoom"],
        contrast=ARGS.VanillaModel["contrast"],
        flip=ARGS.VanillaModel["flip"],
    )
    train_model(
        model_name="VanillaModel",
        model=vanilla_model
    )


def train_resnet50():
    res_net_50_model = mc.get_resnet50_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs['num_classes'],
        resize=ARGS.ResNet50Model["resize"]
    )
    train_model(
        model_name="ResNet50Model",
        model=res_net_50_model
    )


def train_xception():
    xception_model = mc.get_xception_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs['num_classes'],
        resize=ARGS.XceptionModel["resize"]
    )
    train_model(
        model_name="XceptionModel",
        model=xception_model
    )


def train_vgg19():
    vgg19_model = mc.get_vgg19_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs['num_classes'],
        resize=ARGS.VGG19Model["resize"]
    )
    train_model(
        model_name="VGG19Model",
        model=vgg19_model
    )


def train_inceptionresnetv2():
    inception_res_net_v2_model = mc.get_inceptionresnetv2_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs['num_classes'],
        resize=ARGS.InceptionResNetV2Model["resize"]
    )
    train_model(
        model_name="InceptionResNetV2Model",
        model=inception_res_net_v2_model
    )


def train_inceptionv3():
    inception_v3_model = mc.get_inceptionv3_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs['num_classes'],
        resize=ARGS.InceptionV3Model["resize"]
    )
    train_model(
        model_name="InceptionV3Model",
        model=inception_v3_model
    )


def evaluate_model(model_name, test_data, file_names):
    print("Evaluating model {}...".format(model_name))
    optimal_model = tf.keras.models.load_model(
        get_optimal_model(
            ARGS.GlobalArgs["model_dir"] + os.sep + model_name
        )
    )
    result = optimal_model.evaluate(
        test_data,
        verbose=1,
        steps=len(file_names) * ARGS.TFRecordConfig["size"] // ARGS.TestConfig["test_batch_size"] + 1
    )
    print(result)


def main():
    if ARG.mode == "train":
        if ARG.model == "All":
            train_vanilla()
            train_vgg19()
            train_inceptionresnetv2()
            train_resnet50()
            train_xception()
            train_inceptionv3()
        else:
            if ARG.model == "Vanilla":
                train_vanilla()
            elif ARG.model == "ResNet50":
                train_resnet50()
            elif ARG.model == "Xception":
                train_xception()
            elif ARG.model == "VGG19":
                train_vgg19()
            elif ARG.model == "InceptionResNetV2":
                train_inceptionresnetv2()
            elif ARG.model == "InceptionV3":
                train_inceptionv3()
    elif ARG.mode == "test":
        if os.path.isdir(ARG.test):
            file_names = os.listdir(ARG.test)
            file_names = [x for x in file_names if x.endswith('tfrecords') or x.endswith('tfrecord')]
        elif os.path.isfile(ARG.test):
            if ARG.test.endswith('tfrecords') or ARG.test.endswith('tfrecord'):
                file_names = [ARG.test]
            else:
                file_names = []
        else:
            raise FileExistsError("Invalid path.")

        if not file_names:
            raise FileNotFoundError("Can't find files ending with tfrecords/tfrecord.")

        test_data = get_data(
            file_path=file_names,
            buffer_size=ARGS.TestConfig["test_buffer_size"],
            batch_size=ARGS.TestConfig["test_batch_size"],
            auto_tune=ARGS.TestConfig["auto_tune"]
        )

        print("Scanning optimal models from model directory |{}|...".format(ARGS.GlobalArgs["model_dir"]))
        model_files = os.listdir(ARGS.GlobalArgs["model_dir"])
        if model_files:
            print("Found models: {}".format(model_files))
        else:
            raise FileNotFoundError("Model no found, check directory setting of 'model_dir' or train the model first.")

        if ARG.model == "All":
            for model in model_files:
                evaluate_model(
                    model_name=model,
                    test_data=test_data,
                    file_names=file_names
                )
        else:
            evaluate_model(
                model_name="{}Model".format(ARG.model),
                test_data=test_data,
                file_names=file_names
            )
    else:
        raise ValueError("Please specify task from [train, test].")


if __name__ == "__main__":
    mc = ModelCollection()
    ARG = model_parser()
    main()

