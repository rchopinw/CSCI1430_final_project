from DataProcessing import TFRecordData
from ARGS import ARGS

if __name__ == '__main__':
    print("Initializing TF Record Data Writer...")
    tfr = TFRecordData(
        directory=ARGS.TFRecordConfig["directory"],
        save_directory=ARGS.TFRecordConfig["save_directory"],
        size=ARGS.TFRecordConfig["size"],
        image_size=ARGS.TFRecordConfig["image_size"]
    )
    print("TF Record Data Writer Initialized Successfully...")
    print("Converting and Writing...")
    tfr.convert()
    print("Done!")