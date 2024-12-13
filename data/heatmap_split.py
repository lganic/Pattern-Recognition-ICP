import logging
from tqdm import tqdm
import pandas as pd
import os
import random
import shutil
import sys
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("dataset_split.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        logging.info("Starting dataset splitting process.")

        all_files_dir = 'heatmaps'

        TRAIN = 80
        VAL = 18
        TEST = 2

        # Validate split ratios
        if TRAIN + VAL + TEST != 100:
            logging.error("Sum of TRAIN, VAL, and TEST ratios is not 100. Please check your configuration.")
            raise ValueError('Check your math')

        logging.info(f"Split ratios set to TRAIN: {TRAIN}%, VAL: {VAL}%, TEST: {TEST}%.")

        # Check if the dataset path exists
        if not os.path.exists(all_files_dir):
            logging.error(f"The specified dataset path does not exist: {all_files_dir}")
            raise FileNotFoundError(f"The path {all_files_dir} does not exist.")

        all_files = glob(os.path.join(all_files_dir, '*.npy'))

        n_files = len(all_files)
        if n_files == 0:
            logging.warning(f"No files found. Skipping.")
            return

        n_train = TRAIN * n_files // 100
        n_val = VAL * n_files // 100
        n_test = n_files - n_train - n_val

        logging.info(f"Split counts: TRAIN={n_train}, VAL={n_val}, TEST={n_test}.")

        # Create a list indicating the set each file belongs to
        locations = [0] * n_train + [1] * n_val + [2] * n_test
        random.shuffle(locations)
        logging.info(f"Shuffled locations")

        # Define directory paths
        test_location = os.path.join(all_files_dir, 'test')
        train_location = os.path.join(all_files_dir, 'train')
        val_location = os.path.join(all_files_dir, 'val')

        # Create directories if they don't exist
        os.makedirs(test_location, exist_ok=True)
        os.makedirs(train_location, exist_ok=True)
        os.makedirs(val_location, exist_ok=True)
        logging.info(f"Ensured existence of train, val, and test directories.")

        for index, filename in enumerate(tqdm(all_files)):

            file_location = filename

            # Check if the file exists before moving
            if not os.path.isfile(file_location):
                logging.warning(f"File not found: {file_location}. Skipping.")
                continue

            file_set = locations[index]
            image_location = file_location.replace('_heatmap.npy', '.jpg')

            try:
                if file_set == 0:
                    shutil.move(file_location, train_location)
                    shutil.move(image_location, train_location)
                    # shutil.copy(file_location, train_location)
                    logging.debug(f"Moved {filename} to train.")
                elif file_set == 1:
                    shutil.move(file_location, val_location)
                    shutil.move(image_location, val_location)
                    # shutil.copy(file_location, val_location)
                    logging.debug(f"Moved {filename} to val.")
                else:
                    shutil.move(file_location, test_location)
                    shutil.move(image_location, test_location)
                    # shutil.copy(file_location, test_location)
                    logging.debug(f"Moved {filename} to test.")
            except Exception as e:
                logging.error(f"Failed to move {filename}: {e}")

        logging.info("Dataset splitting process completed successfully.")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
