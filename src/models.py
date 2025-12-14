import numpy as np
from src.data_processing import detect_missing_mask


def manual_ordinal_encode(arr, map_dict):
    """Ánh xạ các giá trị thứ tự (chuỗi) sang số, gán NaN cho các giá trị không hợp lệ."""
    encoded = np.array([map_dict.get(val, np.nan) for val in arr], dtype=float)
    return encoded

def get_outlier_mask_iqr(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (arr >= lower_bound) & (arr <= upper_bound)

def independent_ttest_numpy(data1, data2):
    n1 = len(data1)
    n2 = len(data2)
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    var1 = np.var(data1, ddof=1) 
    var2 = np.var(data2, ddof=1)

    # T-statistic 
    numerator = mean1 - mean2
    denominator = np.sqrt((var1 / n1) + (var2 / n2))
    t_stat = numerator / denominator

    # Bậc tự do 
    df_numerator = ((var1 / n1) + (var2 / n2))**2
    df_denominator = ((var1 / n1)**2 / (n1 - 1)) + ((var2 / n2)**2 / (n2 - 1))
    df = df_numerator / df_denominator
    
    return t_stat, df

def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    
    # 1. Tạo chỉ mục ngẫu nhiên
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 2. Tính toán kích thước tập kiểm tra
    n_test = int(n_samples * test_size)
    
    # 3. Tách chỉ mục
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # 4. Trích xuất dữ liệu
    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    y_train = y[train_indices, :]
    y_test = y[test_indices, :]
    
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def chi_square_test_numpy(feature_col, target_col):
    
    # Loại bỏ tạm thời các giá trị 'Unknown' (hoặc Missing) cho mục đích kiểm định
    valid_mask = (feature_col != 'Unknown')
    feature_col = feature_col[valid_mask]
    target_col = target_col[valid_mask]
    
    # 1. Lấy các nhãn duy nhất
    unique_features = np.unique(feature_col)
    unique_targets = np.unique(target_col)
    
    # 2. Lập Bảng Tần suất Quan sát (Observed Frequency)
    observed = np.zeros((len(unique_features), len(unique_targets)))
    
    for i, f_val in enumerate(unique_features):
        for j, t_val in enumerate(unique_targets):
            observed[i, j] = np.sum((feature_col == f_val) & (target_col == t_val))
            
    # 3. Tính Bảng Tần suất Kỳ vọng (Expected Frequency)
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total_samples = observed.sum()
    
    expected = np.outer(row_totals, col_totals) / total_samples
    
    # 4. Tính trị số Chi-square
    chi_sq_stat = np.sum((observed - expected)**2 / (expected + 1e-9))
    
    # 5. Tính Bậc tự do (Degrees of Freedom)
    df = (len(unique_features) - 1) * (len(unique_targets) - 1)
    
    return chi_sq_stat, df, observed 


class LogisticRegressionNumPy:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Hàm Sigmoid."""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Huấn luyện mô hình bằng Gradient Descent."""
        n_samples, n_features = X.shape
        y = y.flatten()
        
        # Khởi tạo tham số (weights, bias)
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # Tính toán đầu ra tuyến tính
            linear_model = np.dot(X, self.weights) + self.bias

            # Tính toán xác suất dự đoán (Hypothesis)
            y_predicted = self._sigmoid(linear_model)
            
            # Tính Gradient (Lỗi)
            error = y_predicted - y
            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)
            
            # Cập nhật Tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Tính Cost (Hàm mất mát Binary Cross-Entropy)
            if i % 100 == 0:
                cost = (-1/n_samples) * np.sum(y * np.log(y_predicted + 1e-9) + (1 - y) * np.log(1 - y_predicted + 1e-9))
                self.cost_history.append(cost)

    def predict_proba(self, X):
        """Dự đoán xác suất."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        """Dự đoán nhãn (0 hoặc 1)."""
        y_predicted = self.predict_proba(X)
        y_predicted_cls = (y_predicted >= 0.5).astype(int)
        return y_predicted_cls
    
def accuracy_score_numpy(y_true, y_pred):
    return np.sum(y_true.flatten() == y_pred) / len(y_true)

def precision_recall_f1_numpy(y_true, y_pred):
    """Tính TN, TP, FN, FP, Precision, Recall, F1-Score."""
    y_true = y_true.flatten()
    
    # Tính các giá trị cơ bản (Confusion Matrix Components)
    TP = np.sum((y_pred == 1) & (y_true == 1)) # True Positive
    FP = np.sum((y_pred == 1) & (y_true == 0)) # False Positive
    FN = np.sum((y_pred == 0) & (y_true == 1)) # False Negative
    TN = np.sum((y_pred == 0) & (y_true == 0)) # True Negative
    
    # Precision, Recall, F1-Score
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    # Trả về tất cả các giá trị
    return TN, TP, FN, FP, precision, recall, f1_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred, name):
    """Tính toán và hiển thị các độ đo sử dụng hàm chuẩn của Sklearn."""
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def calculate_roc_auc_numpy(y_true, y_proba):
    """Tính toán các điểm trên đường cong ROC (FPR, TPR) và AUC chỉ dùng NumPy."""
    # Lấy 100 ngưỡng từ 1.0 đến 0.0. Thứ tự này giúp FPR tăng dần từ 0 đến 1.
    thresholds = np.linspace(1.0, 0.0, 100) 
    tprs = [] # True Positive Rate
    fprs = [] # False Positive Rate
    
    y_true_flat = y_true.flatten()
    
    # Lặp qua các ngưỡng
    for thresh in thresholds:
        # Dự đoán nhãn dựa trên ngưỡng
        y_pred = (y_proba >= thresh).astype(int).flatten()
        
        # Tính toán các thành phần của Confusion Matrix
        TP = np.sum((y_pred == 1) & (y_true_flat == 1))
        FN = np.sum((y_pred == 0) & (y_true_flat == 1))
        FP = np.sum((y_pred == 1) & (y_true_flat == 0))
        TN = np.sum((y_pred == 0) & (y_true_flat == 0))
        
        # Tính TPR (Recall) và FPR (1 - Specificity)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tprs.append(TPR)
        fprs.append(FPR)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    
    
    # Tính AUC bằng quy tắc hình thang (np.trapz)
    auc = np.trapezoid(tprs, fprs)
    
    return fprs, tprs, auc