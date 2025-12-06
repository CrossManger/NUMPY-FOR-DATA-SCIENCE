import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data_processing import compute_rate_by_header, detect_missing_mask

def visualize_multiple_bar_charts(charts, rows, cols, figsize=(12, 8)):
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    axs = axs.ravel()
    
    for i, chart in enumerate(charts):
        cate, values, title, ylabel = chart
        ax = axs[i]

        ax.bar(cate, values)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        
    plt.tight_layout()
    plt.show()

def visualize_target(data, header):
    target_col = data[:, header.index('target')].astype(float)
    total_count = len(target_col)
    target_1_count = np.sum(target_col == 1)
    target_0_count = np.sum(target_col == 0)

    print("\n--- Phân tích Target Imbalance ---\n")
    print(f"Tổng mẫu: {total_count}")
    print(f"Số người muốn thay đổi (Target = 1): {target_1_count} ({target_1_count/total_count:.2%})")

    plt.figure(figsize=(7, 7))
    plt.pie([target_0_count, target_1_count], 
            labels=['Target = 0 (Không đổi)', 'Target = 1 (Muốn đổi)'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=['#66b3ff', '#ff9999'], 
            explode=(0, 0.1))
    plt.title('Tỷ lệ phân bố Target')
    plt.show()

def visualize_categorical(data, header, categorical_cols):
    print("\n--- Phân tích tỷ lệ thay đổi công việc theo Đặc trưng ---")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
    fig.suptitle('Tỷ lệ ứng viên thay đổi công việc (Target = 1) theo các Đặc trưng', fontsize=16)

    target_col = data[:, header.index('target')].astype(float)
    total_count = len(target_col)
    target_1_count = np.sum(target_col == 1)
    avg_target_rate = target_1_count/total_count 

    for i, colname in enumerate(categorical_cols):
        
        row = i // 3  
        col = i % 3   
        
        current_ax = axes[row, col]
            
        unique_val, rates = compute_rate_by_header(data, header, colname)
        
        sort_indices = np.argsort(rates)[::-1]
        unique_val = unique_val[sort_indices]
        rates = rates[sort_indices]

        sns.barplot(x=unique_val, y=rates, hue=unique_val, ax=current_ax, palette='viridis', legend=False)
        
        current_ax.set_title(colname)
        current_ax.set_ylabel('Tỷ lệ Target = 1')
        current_ax.tick_params(axis='x', rotation=45)
        current_ax.axhline(y=avg_target_rate, color='r', linestyle='--', label='Tỷ lệ trung bình')
        current_ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_trend(data, header, trend):
    target_col = data[:, header.index('target')].astype(float)

    experience_col = data[:, header.index(trend)]

    experience_cleaned = np.array([
        '21' if x == '>20' else '0' if x == '<1' else x 
        for x in experience_col
    ])

    missing_mask = detect_missing_mask(experience_cleaned)
    experience_numeric = experience_cleaned[~missing_mask].astype(float)
    target_for_exp = target_col[~missing_mask]

    bins = np.array([0, 5, 10, 15, 20, 100]) # Định nghĩa các nhóm: 0-5, 6-10, 11-15, 16-20, >20
    binned_exp = np.digitize(experience_numeric, bins)

    unique_bins = np.unique(binned_exp)
    binned_rates = []
    bin_labels = []

    for b in unique_bins:
        mask = binned_exp == b
        rate = target_for_exp[mask].mean()
        binned_rates.append(rate)
        
        if b == 1: bin_labels.append('0-5')
        elif b == 2: bin_labels.append('6-10')
        elif b == 3: bin_labels.append('11-15')
        elif b == 4: bin_labels.append('16-20')
        else: bin_labels.append('>20')
        
    plt.figure(figsize=(10, 6))
    plt.plot(bin_labels, binned_rates, marker='o', linestyle='-', color='indigo')
    plt.title('Xu hướng tỷ lệ thay đổi công việc theo nhóm Kinh nghiệm')
    plt.xlabel('Nhóm Kinh nghiệm (Năm)')
    plt.ylabel('Tỷ lệ Muốn thay đổi (Target = 1)')
    plt.grid(True)
    plt.show()

def visualize_city_index(data, header):
    city_dev_index_col = data[:, header.index('city_development_index')]
    target_col = data[:, header.index('target')].astype(float)

    missing_mask_dev = detect_missing_mask(city_dev_index_col)
    city_dev_index_numeric = city_dev_index_col[~missing_mask_dev].astype(float)
    target_for_dev = target_col[~missing_mask_dev]

    group_0_dev = city_dev_index_numeric[target_for_dev == 0]
    group_1_dev = city_dev_index_numeric[target_for_dev == 1]

    print("\n--- Phân tích City Development Index ---")
    print(f"Median Target = 0: {np.median(group_0_dev):.4f}")
    print(f"Median Target = 1: {np.median(group_1_dev):.4f}")

    plt.figure(figsize=(10, 6))
    plt.boxplot([group_0_dev, group_1_dev], 
                tick_labels=['Target = 0 (Không đổi)', 'Target = 1 (Muốn đổi)'])
    plt.title('So sánh City Development Index giữa hai nhóm Target')
    plt.ylabel('City Development Index')
    plt.grid(True)
    plt.show()

def visualize_training_hours(data, header):
    training_score_col = data[:, header.index('training_hours')]
    target_col = data[:, header.index('target')].astype(float)
    
    missing_mask_score = detect_missing_mask(training_score_col)
    training_score_numeric = training_score_col[~missing_mask_score].astype(float)
    target_for_score = target_col[~missing_mask_score]

    group_0_score = training_score_numeric[target_for_score == 0]
    group_1_score = training_score_numeric[target_for_score == 1]

    plt.figure(figsize=(12, 6))

    sns.histplot(group_0_score, label='Target = 0 (Không đổi)', kde=True, color='blue', alpha=0.5)
    sns.histplot(group_1_score, label='Target = 1 (Muốn đổi)', kde=True, color='red', alpha=0.5)

    plt.title('Phân bố số giờ đào tạo giữa hai nhóm Target')
    plt.xlabel('Số giờ đào tạo')
    plt.legend()
    plt.show()

def visualize_analyze_training_hours(bin_labels, binned_rates):
    plt.figure(figsize=(10, 6))
    plt.plot(bin_labels, binned_rates, marker='o', linestyle='-', color='teal')
    plt.title('Xu hướng tỷ lệ thay đổi công việc theo số giờ đào tạo')
    plt.xlabel('Nhóm giờ Đào tạo')
    plt.ylabel('Tỷ lệ Muốn thay đổi (Target = 1)')
    plt.grid(True)
    plt.show()


def visualize_company_features(data, header, compute_rate_by_header):
    company_features = ['company_size', 'company_type', 'relevent_experience']
    
    print("\n--- Phân tích Đặc điểm Công ty & Turnover ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) 
    fig.suptitle('Tỷ lệ ứng viên muốn thay đổi công việc (Target = 1) theo Đặc điểm Công ty', fontsize = 16)

    target_col = data[:, header.index('target')].astype(float)
    total_count = len(target_col)
    target_1_count = np.sum(target_col == 1)
    avg_target_rate = target_1_count/total_count 


    for i, colname in enumerate(company_features):
        current_ax = axes[i]
            
        unique_val, rates = compute_rate_by_header(data, header, colname)
        
        sort_indices = np.argsort(rates)[::-1]
        unique_val = unique_val[sort_indices]
        rates = rates[sort_indices]

        sns.barplot(x=unique_val, y=rates, ax=current_ax) 
        
        current_ax.set_title(colname)
        current_ax.set_ylabel('Tỷ lệ Target = 1')
        current_ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()