import os


# 1. DIRECTORY AND FILE PATHS
BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
MODEL_SAVE_DIR = "models_transformer_current"

# Original Data (Will be dynamically generated if missing)
INPUT_DATA_FILE = os.path.join(BASE_DATA_PATH, "data_merge_final.xlsx")
INPUT_DATA_SHEET = "data"

# Mapping File (Required for remote dataset construction)
MAPPING_ORIGINAL_FILE = os.path.join(BASE_DATA_PATH, "mapping_original.csv")

# Augmentation Data
MASTER_AUGMENTATION_FILE = os.path.join(BASE_DATA_PATH, "master_augmentation.xlsx")
MASTER_AUGMENTATION_SHEET = "Immoral_Threat"

# Lexicon
KAMUS_ALAY_FILE = os.path.join(BASE_DATA_PATH, "new_kamusalay.csv")

# Preprocessed Caching Paths
PREPROCESSED_DATA_FILE = os.path.join(BASE_DATA_PATH, "preprocessed_data_merge_transformer.xlsx")
PREPROCESSED_AUGMENTATION_FILE = os.path.join(BASE_DATA_PATH, "preprocessed_master_augmentation_transformer.xlsx")


# 1b. REMOTE DATA SOURCES (GITHUB)

GITHUB_SOURCE_ALFINA = "https://raw.githubusercontent.com/ialfina/id-hatespeech-detection/f1139d543db69539e05b05ec3d290a6421dc7dd7/IDHSD_RIO_unbalanced_713_2017.txt"

GITHUB_SOURCE_IBROHIM = "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/475b39955c456f92c0b9cd5008613f9198670b94/re_dataset.csv"

GITHUB_SOURCE_PRATIWI = "https://raw.githubusercontent.com/nurindahpratiwi/dataset-hate-speech-instagram/34d13a6f553216851f789dee584b8916451dfbae/572-hate-speech-dataset.csv"


# 2. EXPERIMENTAL PIPELINE FLAGS
USE_HIERARCHICAL = True
USE_UNDERSAMPLING = True
USE_AUGMENTATION = True

UNDERSAMPLING_PERCENTAGE = 0.5  # Only applicable if USE_UNDERSAMPLING is True
USE_PREPROCESSING = True #mandatory for transformer-based models, but can be disabled for further ablation studies

# 3. GLOBAL VARIABLES & METADATA
# Reproducibility 0 42 1337 2026 3407
RANDOM_SEED = 0  

# Data Splitting
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Original Data Column Mapping
TEXT_COLUMN_INPUT = "Data"
ORTHO_FEATURES = ["ortho_capitals_ratio", "ortho_all_caps_words"]
LABEL_COLUMN_INPUT = "Jenis Ujaran"
BINARY_TARGET_LABEL = "neutral"
PARENT_ID_INPUT = "parent_id"

# Augmentation Data Column Mapping
TEXT_COLUMN_AUGMENTATION = "Augmented Data"
LABEL_COLUMN_AUGMENTATION = "Jenis Ujaran"
PARENT_ID_AUGMENTATION = "parent_id"
CHILD_ID_AUGMENTATION = "child_id"


# 4. EXPORT AND METADATA CONFIGURATION
MODEL_VERSION = f"tf{'_aug' if USE_AUGMENTATION else ''}{'_hierarchical' if USE_HIERARCHICAL else 'flat'}{'_undersampling' if USE_UNDERSAMPLING else ''}{'_preprocessing' if USE_PREPROCESSING else ''}_s{RANDOM_SEED}"

MODEL_EXPORT_DIR = os.path.join(MODEL_SAVE_DIR, MODEL_VERSION)
THRESHOLD_FILE = "best_threshold.json"
METADATA_FILE = "experiment_metadata.json"


# 5. HYPERPARAMETERS (INDOBERT)
MODEL_CHECKPOINT = "indobenchmark/indobert-base-p1"
MAX_LEN = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 5