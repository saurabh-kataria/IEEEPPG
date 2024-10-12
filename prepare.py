import numpy as np
from scipy.signal import resample

# Function to downsample the signal
def downsample_signal(ppg_signal, original_fs=125, target_fs=40):
    num_samples = int(len(ppg_signal) * target_fs / original_fs)
    return resample(ppg_signal, num_samples)

# Function to load the dataset and process it
def load_ts_file_with_hr(file_path):
    data = []
    hr_data = []
    start_processing = False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '@data' in line:
                start_processing = True
                continue
            if not start_processing or not line:
                continue

            parts = line.split(':')
            if len(parts) < 2:
                print(f"Error parsing line: {line[:60]}... Error: Line has fewer than expected parts")
                continue

            # Collect the full PPG signal 1
            ppg_signal = [float(x) for x in parts[0].split(',')]
            # Take the second part as the heart rate
            heart_rate = float(parts[-1].strip())
            data.append(ppg_signal)  # Store full PPG signal 1
            hr_data.append(heart_rate)

    return np.array(data), np.array(hr_data)

# Prepare the data with downsampling
def prepare_data(file_path, original_fs=125, target_fs=40):
    ppg_signal_1, heart_rates = load_ts_file_with_hr(file_path)

    # Downsample the PPG signal 1
    downsampled_ppg = [downsample_signal(ppg, original_fs, target_fs) for ppg in ppg_signal_1]

    # Return downsampled PPG signals and heart rates
    return np.array(downsampled_ppg), np.array(heart_rates)

# Process the training and testing datasets
def process_ts_dataset(train_file, test_file, original_fs=125, target_fs=40):
    # Prepare the training data
    X_train, y_train = prepare_data(train_file, original_fs=original_fs, target_fs=target_fs)

    # Prepare the test data
    X_test, y_test = prepare_data(test_file, original_fs=original_fs, target_fs=target_fs)

    return X_train, X_test, y_train, y_test

# File paths for the train and test datasets
train_file = 'IEEEPPG_TRAIN.ts'
test_file = 'IEEEPPG_TEST.ts'

# Process the dataset and get train and test sets
X_train, X_test, y_train, y_test = process_ts_dataset(train_file, test_file)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')

np.save('X_IEEEPPG_train.npy', X_train)
np.save('X_IEEEPPG_test.npy', X_test)
np.save('y_IEEEPPG_train.npy', y_train)
np.save('y_IEEEPPG_test.npy', y_test)

'''
Shape of X_train: (1768, 320)
Shape of X_test: (1328, 320)
Shape of y_train: (1768,)
Shape of y_test: (1328,)
'''
