import numpy as np

def load_data(file_path, delimiter=',', dtype=None, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        header = f.readline().strip().split(delimiter)
    data = np.genfromtxt(file_path, delimiter=delimiter, dtype=str, skip_header=1, autostrip=True)
    return header, data


def detect_missing_mask(arr, missing_values=['', 'NA', 'NaN', 'nan', 'null', 'None']):
    mask = np.zeros(arr.shape, dtype=bool)
    for miss in missing_values:
        mask = mask | (arr == miss)
    mask = mask | (np.char.strip(arr) == '')
    return mask


def compute_rate_by_header(data, header, colname):
    col_idx = header.index(colname)
    target_idx = header.index("target")

    col = data[:, col_idx]
    target = data[:, target_idx].astype(float)

    # Bỏ missing ("")
    mask = col != ""
    col = col[mask]
    target = target[mask]

    unique_val = np.unique(col)
    rates = []
    
    for val in unique_val:
        mask = col == val
        rate = target[mask].mean()
        rates.append(rate)

    return unique_val, np.array(rates)

def analyze_training_hours(data, header):
    target_col = data[:, header.index('target')].astype(float)    
    hours_col = data[:, header.index('training_hours')]

    missing_mask = detect_missing_mask(hours_col)
    hours_numeric = hours_col[~missing_mask].astype(float)
    target_for_hours = target_col[~missing_mask]

    bins = np.array([0, 50, 100, 200, 1000]) 
    binned_hours = np.digitize(hours_numeric, bins)

    unique_bins = np.unique(binned_hours)
    binned_rates = []
    bin_labels = []

    # Tính tỷ lệ Target=1 cho từng nhóm (CHỈ DÙNG NUMPY)
    for b in unique_bins:
        mask = binned_hours == b
        rate = target_for_hours[mask].mean()
        binned_rates.append(rate)
        
        # Gán nhãn cho bin
        if b == 1: bin_labels.append('0-50 giờ')
        elif b == 2: bin_labels.append('51-100 giờ')
        elif b == 3: bin_labels.append('101-200 giờ')
        else: bin_labels.append('>200 giờ')

    return bin_labels, binned_rates