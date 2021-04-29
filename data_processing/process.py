import yaml

from pathlib import Path
from data_processing.DataProcessing import DataWriter, DataReader


def main(config):

    # Convert dataset and write the processed files.
    bucket_name = config["bucket_name"]  # Name of how bucket is mounted
    set_type = "val"
    writer = DataWriter(bucket_name, config, set_type)

    # writer.process_files()
    # writer.write_files()

    # Read dataset.
    set_type = "val"
    base_path = "DataSet/cityscape/processed_data"
    reader = DataReader(base_path, set_type)
    # reader.read_data_set()
    # data_set = reader.get_dataset()


if __name__ == "__main__":
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)

    main(config)
