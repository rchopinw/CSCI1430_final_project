import os
from DataProcessing import train_validation_split, get_data, get_generator
from Models import (
    get_vgg_19_model,
    get_vanilla_model,
    get_xception_model,
    get_res_net_50_model,
    get_inception_v3_model,
    get_inception_res_net_v2_model
)
from ARGS import ARGS
import argparse


def model_parser():
    parser = argparse.ArgumentParser(
        description="Face Mask Detection Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=['Vanilla', "ResNet50", "Xception", "VGG19", "InceptionV3", "InceptionResNetV2"],
        help="Please select model from [Vanilla, ResNet50, Xception, VGG19, Inception] to train."
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

    train_generator = get_generator(train_data)
    validation_generator = get_generator(validation_data)

    model.compile(
        optimizer=ARGS.__dict__[model_name]["train_optimizer"],
        loss=ARGS.__dict__[model_name]["train_loss"],
        metrics=ARGS.__dict__[model_name]["metrics"]
    )

    steps_per_epoch = len(train_files) * ARGS.TFRecordConfig["size"] // ARGS.__dict__[model_name]["train_batch_size"] + 1
    validation_steps = \
        len(validation_files) * ARGS.TFRecordConfig["size"] // ARGS.__dict__[model_name]["validation_batch_size"] + 1

    model.fit(
        x=train_generator,
        validation_data=validation_generator,
        epochs=ARGS.__dict__[model_name]["train_epoch"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=ARGS.__dict__[model_name]["callbacks"]
    )


def main():
    if ARG.model == "Vanilla":
        vanilla_model = get_vanilla_model(
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
    elif ARG.model == "ResNet50":
        res_net_50_model = get_res_net_50_model(
            input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
            num_classes=ARGS.GlobalArgs['num_classes'],
            resize=ARGS.ResNet50Model["resize"]
        )
        train_model(
            model_name="ResNet50Model",
            model=res_net_50_model
        )
    elif ARG.model == "Xception":
        xception_model = get_xception_model(
            input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
            num_classes=ARGS.GlobalArgs['num_classes'],
            resize=ARGS.XceptionModel["resize"]
        )
        train_model(
            model_name="XceptionModel",
            model=xception_model
        )
    elif ARG.model == "VGG19":
        vgg19_model = get_vgg_19_model(
            input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
            num_classes=ARGS.GlobalArgs['num_classes'],
            resize=ARGS.VGG19Model["resize"]
        )
        train_model(
            model_name="VGG19Model",
            model=vgg19_model
        )
    elif ARG.model == "InceptionResNetV2":
        inception_res_net_v2_model = get_inception_res_net_v2_model(
            input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
            num_classes=ARGS.GlobalArgs['num_classes'],
            resize=ARGS.InceptionResNetV2Model["resize"]
        )
        train_model(
            model_name="InceptionResNetV2Model",
            model=inception_res_net_v2_model
        )
    elif ARG.model == "InceptionV3":
        inception_v3_model = get_inception_v3_model(
            input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
            num_classes=ARGS.GlobalArgs['num_classes'],
            resize=ARGS.InceptionV3Model["resize"]
        )
        train_model(
            model_name="InceptionV3Model",
            model=inception_v3_model
        )


if __name__ == "__main__":
    ARG = model_parser()
    main()

