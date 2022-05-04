from DataProcessing import TFRecordData
from ARGS import ARGS

if __name__ == '__main__':
    # print("Initializing TF Record Data Writer...")
    # train_validation_tfr = TFRecordData(
    #     directory=ARGS.TFRecordConfig["directory"],
    #     save_directory=ARGS.TFRecordConfig["save_directory"],
    #     size=ARGS.TFRecordConfig["size"],
    #     image_size=ARGS.TFRecordConfig["image_size"]
    # )
    # print("TF Record Data Writer Initialized Successfully...")
    # print("Converting and Writing...")
    # train_validation_tfr.convert()
    # print("Done!")

    print("Configuring test data...")
    test_tfr = TFRecordData(
        directory=ARGS.TFRecordConfig["test_directory"],
        save_directory=ARGS.TFRecordConfig["test_save_directory"],
        size=ARGS.TFRecordConfig["size"],
        image_size=ARGS.TFRecordConfig["image_size"]
    )
    test_tfr.convert()
    print("Done!")