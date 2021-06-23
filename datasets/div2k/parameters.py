class Div2kParameters:
    def __init__(self, dataset_key, save_data_directory):
        if dataset_key not in available_datasets.keys():
            raise ValueError(f"available datasets are: {available_datasets.keys()}")

        dataset_parameters = available_datasets[dataset_key]

        self.train_directory = dataset_parameters["train_directory"]
        self.train_url = dataset_parameters["train_url"]
        self.valid_directory = dataset_parameters["valid_directory"]
        self.valid_url = dataset_parameters["valid_url"]
        self.scale = dataset_parameters["scale"]

        self.save_data_directory = save_data_directory


available_datasets = {
    "bicubic_x2": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
        "train_directory": "DIV2K_train_LR_bicubic/X2",
        "valid_directory": "DIV2K_valid_LR_bicubic/X2",
        "scale": 2
    },
    "unknown_x2": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip",
        "train_directory": "DIV2K_train_LR_unknown/X2",
        "valid_directory": "DIV2K_valid_LR_unknown/X2",
        "scale": 2
    },
    "bicubic_x3": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip",
        "train_directory": "DIV2K_train_LR_bicubic/X3",
        "valid_directory": "DIV2K_valid_LR_bicubic/X3",
        "scale": 3
    },
    "unknown_x3": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip",
        "train_directory": "DIV2K_train_LR_unknown/X3",
        "valid_directory": "DIV2K_valid_LR_unknown/X3",
        "scale": 3
    },
    "bicubic_x4": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
        "train_directory": "DIV2K_train_LR_bicubic/X4",
        "valid_directory": "DIV2K_valid_LR_bicubic/X4",
        "scale": 4
    },
    "unknown_x4": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip",
        "train_directory": "DIV2K_train_LR_unknown/X4",
        "valid_directory": "DIV2K_valid_LR_unknown/X4",
        "scale": 4
    },
    "realistic_mild_x4": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_mild.zip",
        "train_directory": "DIV2K_train_LR_mild",
        "valid_directory": "DIV2K_valid_LR_mild",
        "scale": 4
    },
    "realistic_difficult_x4": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_difficult.zip",
        "train_directory": "DIV2K_train_LR_difficult",
        "valid_directory": "DIV2K_valid_LR_difficult",
        "scale": 4
    },
    "realistic_wild_x4": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_wild.zip",
        "train_directory": "DIV2K_train_LR_wild",
        "valid_directory": "DIV2K_valid_LR_wild",
        "scale": 4
    },
    "bicubic_x8": {
        "train_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip",
        "valid_url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip",
        "train_directory": "DIV2K_valid_LR_x8",
        "valid_directory": "DIV2K_train_LR_x8",
        "scale": 8
    }
}
