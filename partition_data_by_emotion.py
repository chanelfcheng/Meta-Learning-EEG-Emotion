import os
import numpy as np
import pickle

def load_npz_data(eeg_dir, eye_dir, file_name, cv_number):
    eeg_data_pickle = np.load( os.path.join(eeg_dir, file_name))
    eye_data_pickle = np.load( os.path.join(eye_dir, file_name))
    eeg_data = pickle.loads(eeg_data_pickle['data'])
    eye_data = pickle.loads(eye_data_pickle['data'])
    label = pickle.loads(eeg_data_pickle['label'])

    print("size of eeg data: ", len(eeg_data))
    print("size of eye data: ", len(eye_data))
    print("size of labels: ", len(label))
    
    list_1 = [0,1,2,3,4,15,16,17,18,19,30,31,32,33,34]
    list_2 = [5,6,7,8,9,20,21,22,23,24,35,36,37,38,39]
    list_3 = [10,11,12,13,14,25,26,27,28,29,40,41,42,43,44]
    if cv_number == 1:
        print('#1 as test, preparing data')
        train_list = list_2 + list_3
        test_list = list_1
    elif cv_number == 2:
        print('#2 as test, preparing data')
        train_list = list_1 + list_3
        test_list = list_2
    else:
        print('#3 as test, preparing data')
        train_list = list_1 + list_2
        test_list = list_3

    train_eeg = []
    test_eeg = []
    train_label = []
    for session_id in range(len(train_list)):
        train_eeg_tmp = eeg_data[train_list[session_id]]
        train_eye_tmp = eye_data[train_list[session_id]]
        train_label_tmp = label[train_list[session_id]]
        if session_id == 0:
            train_eeg = train_eeg_tmp
            train_eye = train_eye_tmp
            train_label = train_label_tmp
        else:
            train_eeg = np.vstack((train_eeg, train_eeg_tmp))
            train_eye = np.vstack((train_eye, train_eye_tmp))
            train_label = np.hstack((train_label, train_label_tmp))    
        # print(eeg_data[train_list[session_id]].shape)
        # print(eeg_data[train_list[session_id]].shape)
        # input("Press enter to continue...")
    assert train_eeg.shape[0] == train_eye.shape[0]
    assert train_eeg.shape[0] == train_label.shape[0]

    test_eeg = []
    test_eye = []
    test_label = []
    for test_id in range(len(test_list)):
        test_eeg_tmp = eeg_data[test_list[test_id]]
        test_eye_tmp = eye_data[test_list[test_id]]
        test_label_tmp = label[test_list[test_id]]
        if test_id == 0:
            test_eeg = test_eeg_tmp
            test_eye = test_eye_tmp
            test_label = test_label_tmp
        else:
            test_eeg = np.vstack((test_eeg, test_eeg_tmp))
            test_eye = np.vstack((test_eye, test_eye_tmp))
            test_label = np.hstack((test_label, test_label_tmp))
    assert test_eeg.shape[0] == test_eye.shape[0]
    assert test_eeg.shape[0] == test_label.shape[0]

    train_all = np.hstack((train_eeg, train_eye, train_label.reshape([-1,1])))
    test_all = np.hstack((test_eeg, test_eye, test_label.reshape([-1,1])))

    return train_all, test_all

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
    
    # main folder for saving partitioned data
    partition_dir = os.path.join(os.getcwd(), 'datasets/seed-v')

    # dictionary of numerical label : emotion
    emotion_dict = {0: 'disgust', 1: 'fear', 2: 'sad', 3: 'neutral', 4: 'happy'}

    # create emotion folders for saving partitioned data
    for emotion in emotion_dict.values():
        create_if_not_exists(os.path.join(partition_dir, 'support', emotion))
        create_if_not_exists(os.path.join(partition_dir, 'query', emotion))
        
    num = 1
    for file_name in file_list:
        print(file_name)
        train_data, test_data = load_npz_data(eeg_dir, eye_dir, file_name, 3)
        # train_labels, test_labels = train_data[:,-1], test_data[:,-1]

        # start emotion count for support data
        emotion_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for sample in train_data: # save train/support data to their corresponding emotion folders
            # which directory to save the data to based on the emotion label
            label = int(sample[-1])
            save_dir = os.path.join(partition_dir, 'support', emotion_dict[label])
            # concatenate with the filename
            # save the label to where it belongs to            
            save_file = os.path.join(save_dir, f"p{num}_{emotion_count[label]}")
            emotion_count[label] += 1
            np.save(save_file, sample)

        # reset emotion count for query data
        emotion_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for sample in test_data: # do the same thing for the test/query data
            label = sample[-1]
            save_dir = os.path.join(partition_dir, 'query', emotion_dict[label])
            save_file = os.path.join(save_dir, f"p{num}_{emotion_count[label]}")
            emotion_count[label] += 1
            np.save(save_file, sample)
        
        num += 1
        
