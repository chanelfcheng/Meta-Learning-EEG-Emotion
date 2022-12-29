import os
import numpy as np
import pickle

def load_npz_data(eeg_dir, eye_dir, file_name, cv_number):
    eeg_data_pickle = np.load( os.path.join(eeg_dir, file_name))
    eye_data_pickle = np.load( os.path.join(eye_dir, file_name))

    eeg_data_dict = pickle.loads(eeg_data_pickle['data'])
    eeg_data = [j for sub in list(eeg_data_dict.values()) for j in sub]

    eye_data_dict = pickle.loads(eye_data_pickle['data'])
    eye_data = [j for sub in list(eye_data_dict.values()) for j in sub]
    
    label_dict = pickle.loads(eeg_data_pickle['label'])
    labels = np.array([j for sub in list(label_dict.values()) for j in sub])

    print("size of eeg data: ", len(eeg_data))
    print("size of eye data: ", len(eye_data))
    print("size of labels: ", len(labels))

    # there will be thousands of samples for each emotion
    all_data = np.hstack((eeg_data, eye_data, labels.reshape([-1,1])))

    return all_data

def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    eeg_dir = os.path.join(os.getcwd(), 'raw_data/seed-v/eeg_data')
    eye_dir = os.path.join(os.getcwd(), 'raw_data/seed-v/eye_data')
    file_list = os.listdir(eeg_dir) # Both eeg and eye data have the same file names
    
    # main folder for saving merged data
    merge_dir = os.path.join(os.getcwd(), 'raw_data/seed-v/merged_data')
    create_if_not_exists(merge_dir)
    
    for file in file_list:
        print(file)
        all_data = load_npz_data(eeg_dir, eye_dir, file, 3)
        file_name = f"p_{file.split('_')[0]}" # extracts only the participant number
        np.save(os.path.join(merge_dir, file_name), all_data)