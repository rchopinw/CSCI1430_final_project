import os
from DataProcessing import train_validation_split, get_data, get_generator
from Models import get_res_net_50_model, get_vanilla_model, get_xception_model
from ARGS import ARGS


def train_model(model_name, model):
    print("...Training ResNet50 Model...")
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


if __name__ == "__main__":
    # training Vanilla model:
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

    # training ResNet50 model:
    # res_net_50_model = get_res_net_50_model(
    #     input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
    #     num_classes=ARGS.GlobalArgs['num_classes'],
    #     resize=ARGS.ResNet50Model["resize"]
    # )
    # train_model(
    #     model_name="ResNet50Model",
    #     model=res_net_50_model
    # )

    # training Xception Model:
    # xception_model = get_xception_model(
    #     input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
    #     num_classes=ARGS.GlobalArgs['num_classes'],
    #     resize=ARGS.XceptionModel["resize"]
    # )
    # train_model(
    #     model_name="XceptionModel",
    #     model=xception_model
    # )
    