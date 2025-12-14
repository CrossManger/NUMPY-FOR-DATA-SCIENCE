# 🚀 HR ANALYTICS: DỰ ĐOÁN KHẢ NĂNG THAY ĐỔI CÔNG VIỆC CỦA DATA SCIENTIST

Mô hình dự đoán các ứng viên Khoa học Dữ liệu có khả năng tìm kiếm công việc mới, được xây dựng **hoàn toàn bằng NumPy** để đạt hiệu suất và tuân thủ yêu cầu kỹ thuật cao.

---

## 📚 Mục lục

1.  [Giới thiệu và Bài toán](#1-giới-thiệu-và-bài-toán)
2.  [Dataset](#2-dataset)
3.  [Methodology (Phương pháp)](#3-methodology-phương-pháp)
4.  [Installation & Setup](#4-installation--setup)
5.  [Usage (Cách chạy)](#5-usage-cách-chạy)
6.  [Results (Kết quả)](#6-results-kết-quả)
7.  [Project Structure (Cấu trúc dự án)](#7-project-structure-cấu-trúc-dự-án)
8.  [Challenges & Solutions (Thử thách & Giải pháp)](#9-challenges--solutions-thử-thách--giải-pháp)
9.  [Future Improvements (Cải tiến tương lai)](#10-future-improvements-cải-tiến-tương-lai)
10. [Contributors & Contact](#11-contributors--contact)
11. [License](#12-license)

---

## 1. Giới thiệu và Bài toán

### 🎯 Bài toán: Dự đoán Khả năng Đổi việc

Bài toán yêu cầu dự đoán liệu ứng viên có muốn thay đổi công việc hay không.

* **Động lực & Ứng dụng:** Dự đoán **Turnover** giúp bộ phận Nhân sự chủ động giảm thiểu rủi ro mất nhân tài.
* **Mục tiêu cụ thể:** Sử dụng thành thạo NumPy để xử lý dữ liệu và cài đặt thuật toán Logistic Regression từ đầu.

### 🧩 Khám phá Dữ liệu (EDA) theo Định hướng Câu hỏi

Quá trình phân tích tập trung vào 3 câu hỏi chính:

#### 🎯 Câu hỏi 1 --- Phân tích Nhân khẩu học và Kinh nghiệm

* **Những nhóm ứng viên nào có tỷ lệ muốn chuyển việc cao nhất?**
* **Mục tiêu:** Xác định nhóm nguy cơ "turnover" cao nhất.

#### 🎯 Câu hỏi 2 --- Ảnh hưởng của Đào tạo

* **Training\_hours có ảnh hưởng đến việc ứng viên quyết định đổi việc không?**
* **Mục tiêu:** Đánh giá xem liệu việc đầu tư vào đào tạo có giúp giữ chân nhân viên hay không.

#### 🎯 Câu hỏi 3 --- Đặc điểm Công ty & Turnover

* **Các đặc điểm về công ty ảnh hưởng như thế nào đến khả năng đổi việc?**
* **Mục tiêu:** Phân tích các yếu tố từ phía doanh nghiệp tác động đến xu hướng thay đổi việc làm.

---

## 2. Dataset

* **Nguồn dữ liệu:** HR Analytics: Job Change of Data Scientists ([HR Analytics](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)).
* **Kích thước:** Tập Huấn luyện (`aug_train.csv`) có khoảng 19158 hàng.
* **Đặc điểm:** Dữ liệu chứa Missing Values, biến thứ tự/phân loại, và có sự mất cân bằng trong biến mục tiêu.

---

## 3. Methodology (Phương pháp)

Toàn bộ quá trình xử lý và tính toán được thực hiện **CHỈ** sử dụng thư viện NumPy.

### 3.1 Quy trình Xử lý Dữ liệu (Preprocessing)
* **Imputation (Điền thiếu):** Sử dụng $\text{Median}$ hoặc $\text{'Unknown'}$.
* **Outlier Handling:** Sử dụng IQR để xác định và loại bỏ Outlier trên các biến số liên tục.
* **Standardization (Z-score):** Áp dụng $Z$-score ($\mu=0, \sigma=1$) cho các biến số, cần thiết cho thuật toán dựa trên gradient.
* **Feature Engineering:** Tạo các đặc trưng tương tác mạnh mẽ:
    * **Opportunity Gap:** $\text{log}(1 + \text{Experience}) / \text{City Dev Index}$.
    * **Training Ratio:** $\text{Training Hours} / \text{Experience}$.

### 3.2 Thuật toán: Logistic Regression (NumPy Implementation)

Mô hình Logistic Regression được cài đặt từ đầu.

* **Hypothesis (Dự đoán) (Hàm sigmoid):** 
    $$h_{\theta}(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
    
* **Hàm mất mát (Cost Function):** Binary Cross-Entropy.

* **Tối ưu hóa:** **Gradient Descent** sử dụng Vectorization và các phép tính số học tối ưu.

---

## 4. Installation & Setup

1.  Clone repository: `git clone https://github.com/CrossManger/NUMPY-FOR-DATA-SCIENCE.git`
2.  Cài đặt thư viện: `pip install -r requirements.txt`

---

## 5. Usage (Cách chạy)

1.  **Chạy Preprocessing:** Thực thi file `02_preprocessing.ipynb`.
2.  **Chạy Modeling:** Thực thi file `03_modeling.ipynb` để huấn luyện, đánh giá và so sánh mô hình.

---

## 6. Results (Kết quả)

### 6.1 Phân tích và Kết quả Thống kê

| Feature | T-Statistic<br>( $\|T\|$ ) | Ngưỡng Critical<br>($1.96$) | Kết luận |
| :--- | :---: | :---: | :--- |
| **Training Hours** | -1.4138 | $1.96$ | Chưa đủ bằng chứng để bác bỏ **H0: Trung bình Giờ Đào tạo là bằng nhau giữa nhóm Đổi việc và nhóm Không đổi việc.** |
| **Opportunity Gap (FE)** | 3.6382 | $1.96$ | **Opportunity Gap có sự khác biệt có ý nghĩa thống kê giữa hai nhóm.** (Đặc trưng FE có ảnh hưởng lớn). |




### 6.2 So sánh Hiệu suất Mô hình (Test Set)

| Độ Đo | NumPy Custom | Scikit-learn | Phân tích |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7764 | 0.7761 | Mức độ tiệm cận giữa mô hình tự cài đặt và mô hình chuẩn là gần như bằng nhau |
| **F1-Score** | 0.4266 | 0.4342 | F1-Score là độ đo chính xác nhất cho dữ liệu mất cân bằng. |

### 6.3. Đồ thị ROC Curve và AUC

Để đánh giá khả năng phân biệt lớp của mô hình độc lập với ngưỡng cắt xác suất, ta sử dụng đồ thị ROC Curve (Receiver Operating Characteristic Curve) và giá trị AUC (Area Under the Curve).

Kết quả của mô hình tự cài đặt (NumPy Custom LR): **0.7998**


![Đồ thị ROC Curve của NumPy](https://github.com/user-attachments/assets/ca7bc7de-0903-4060-8733-051947348855)

---

## 7. Project Structure (Cấu trúc dự án)

Cấu trúc tuân thủ các yêu cầu kỹ thuật:

```
NUMPY FOR DATA SCIENCE/ 
├── README.md 
├── requirements.txt 
├── data/ 
│ ├── raw/ 
│ └── processed/ 
├── notebooks/ 
│ ├── 01_data_exploration.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_modeling.ipynb 
├── src/ 
│ ├── data_processing.py 
│ ├── model.py
│ └── visualization.py
```

---

## 8. Challenges & Solutions (Thử thách & Giải pháp)

* **Thử thách 1:** Đảm bảo **No Data Leakage** khi xử lý cấu trúc dữ liệu Train/Test riêng biệt.
    * **Giải pháp:** Chia tập train thành 80/20 để tách biệt hoàn toàn việc tiền xử lý dữ liệu (đảm bảo tập validation không bị rò rỉ thông tin).
* **Thử thách 2:** Cài đặt **Logistic Regression** và **Gradient Descent** chỉ dùng NumPy.
    * **Giải pháp:** Sử dụng rộng rãi kỹ thuật **Vectorization** và áp dụng các phép toán số học để đạt được **ổn định số học**.

---

## 9. Future Improvements (Cải tiến tương lai)

* Thử nghiệm kỹ thuật Regularization ($\text{L1}/\text{L2}$) cho mô hình NumPy.
* Áp dụng thuật toán tối ưu hóa nâng cao hơn (ví dụ: $\text{Adam}$) thay vì Gradient Descent cổ điển.

---

## 10. Contributors & Contact

* **Thông tin tác giả:** Vũ Hoàng Minh - 23127427
* **Contact:** vhminh23@clc.fitus.edu.vn

---

## 11. License

Dự án này được phát hành dưới giấy phép **MIT License**.
