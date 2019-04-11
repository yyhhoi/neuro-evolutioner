import pickle
import numpy as np

def load_pickle(file_path):
    with open(file_path, "rb") as fh:
        loaded_file = pickle.load(fh)
    return loaded_file

def write_pickle(file_path, data):
    with open(file_path, "wb") as fh:
        pickle.dump(data, fh)
    return None



class OnlineFilter_np():
    def __init__(self, input_size, kernel_size, time_step):
        self.input_size = input_size
        self.denominator = kernel_size * time_step
        self.kernel_size = kernel_size
        self.input_records = self._create_records_arr()
        self.input_times = 0
    def add(self, input_val):
        assert input_val.shape == self.input_size
        if self.input_times < self.kernel_size:
            self.input_records[self.input_times, ] = input_val
            self.input_times += 1
            return np.nansum(self.input_records[0:self.input_times, ], axis = 0)/self.denominator
        else:
            arr_idx = self.input_times % self.kernel_size
            self.input_records[arr_idx, ] = input_val
            self.input_times += 1
            return np.nansum(self.input_records, axis = 0)/self.denominator
    def get_last(self):
        if self.input_times > 0:
            return self.input_records[-1, ]
    def _create_records_arr(self):
        """
        If input_size = (2, 17), and kernel_size = 15, this function will return zero's array of shape (15, 2, 17)
        Hence, input history will always be stored at axis=0
        """
        records_size = [self.kernel_size]
        for i in range(len(self.input_size)):
            records_size.append(self.input_size[i])
        input_records = np.zeros(records_size)
        return input_records