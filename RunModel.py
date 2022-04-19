from ARGS import ARGS
from DataProcessing import train_validation_split, get_data_generator
import os


if __name__ == "__main__":
    file_path = "/home/bangxi_xiao2/data"
    files = [file_path + '/' + x for x in os.listdir(file_path)]
    print(files)
    train_files, validation_files = train_validation_split(
        file_names=files,
        split_rate=ARGS.GlobalArgs["validation_split"]
    )
    train_generator = get_data_generator(
        file_path=train_files,
        buffer_size=ARGS.ResNet50TrainArgs["train_buffer_size"],
        batch_size=ARGS.ResNet50TrainArgs["train_batch_size"],
        auto_tune=ARGS.ResNet50TrainArgs["auto_tune"]
    )
    validation_generator = ...

    # model part


    

