import os
import tensorflow as tf
from DataProcessing import train_validation_split, get_data, get_generator
from ResNet50Model import ResNet50Model, get_res_net_50_model
from ARGS import ARGS


if __name__ == "__main__":
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
        buffer_size=ARGS.ResNet50TrainArgs["train_buffer_size"],
        batch_size=ARGS.ResNet50TrainArgs["train_batch_size"],
        auto_tune=ARGS.ResNet50TrainArgs["auto_tune"]
    )
    validation_data = get_data(
        file_path=validation_files,
        buffer_size=ARGS.ResNet50TrainArgs["validation_buffer_size"],
        batch_size=ARGS.ResNet50TrainArgs["validation_batch_size"],
        auto_tune=ARGS.ResNet50TrainArgs["auto_tune"]
    )

    train_generator = get_generator(train_data)
    validation_generator = get_generator(validation_data)

    # model = ResNet50Model(
    #     input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
    #     num_classes=ARGS.GlobalArgs['num_classes'],

    model = get_res_net_50_model(
        input_size=(*ARGS.TFRecordConfig['image_size'], ARGS.TFRecordConfig['num_channels']),
        num_classes=ARGS.GlobalArgs['num_classes']
    )

    model.compile(
        optimizer=ARGS.ResNet50TrainArgs["train_optimizer"],
        loss=ARGS.ResNet50TrainArgs["train_loss"],
        metrics=ARGS.ResNet50TrainArgs["metrics"]
    )

    steps_per_epoch = len(train_files) * ARGS.TFRecordConfig["size"] // ARGS.ResNet50TrainArgs["train_batch_size"] + 1
    validation_steps = \
        len(validation_files) * ARGS.TFRecordConfig["size"] // ARGS.ResNet50TrainArgs["validation_batch_size"] + 1

    model.fit(
        x=train_generator,
        validation_data=validation_generator,
        epochs=ARGS.ResNet50TrainArgs["train_epoch"],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=ARGS.ResNet50TrainArgs["callbacks"]
    )