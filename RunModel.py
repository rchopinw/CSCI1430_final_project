from ARGS import ARGS
from DataProcessing import train_validation_split, get_data, get_generator
import os


if __name__ == "__main__":
    file_path = "/home/bangxi_xiao2/data"
    files = [file_path + '/' + x for x in os.listdir(file_path)]
    print(files)
    train_files, validation_files = train_validation_split(
        file_names=files,
        split_rate=ARGS.GlobalArgs["validation_split"]
    )
    train_data = get_data(
        file_path=train_files[:2],
        buffer_size=ARGS.ResNet50TrainArgs["train_buffer_size"],
        batch_size=ARGS.ResNet50TrainArgs["train_batch_size"],
        auto_tune=ARGS.ResNet50TrainArgs["auto_tune"]
    )
    train_generator = get_generator(
        train_data        
    )
    

    for img, label in train_generator:
        print(img.shape, label.shape)
    # model part


    

