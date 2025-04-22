import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load và xử lý dữ liệu
df = pd.read_csv("new_data.csv")
df.rename(columns={'Cở sở hạ tầng': 'Cơ sở hạ tầng'}, inplace=True)

# 2. Mã hóa các cột phân loại với label map rõ ràng
categorical_cols = ['Vị trí', 'Hướng', 'Tiện ích', 'Cơ sở hạ tầng', 'Xu hướng']
encoders = {}

# Tùy chỉnh chú thích cho từng cột phân loại
custom_label_map = {
    'Vị trí': {0: 'Nông thôn', 1: 'Thành thị'},
    'Hướng': {0: 'Bắc', 1: 'Đông', 2: 'Nam', 3: 'Tây'},
    'Tiện ích': {0: 'Thiếu', 1: 'Đầy đủ'},
    'Cơ sở hạ tầng': {0: 'Thiếu', 1: 'Chưa hoàn thiện', 2: 'Đầy đủ'},
    'Xu hướng': {0: 'Đang xuống', 1: 'Ổn định', 2: 'Đang lên'}
}

# Mã hóa + lưu LabelEncoder (phòng trường hợp cần dùng lại)
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Tách đặc trưng và target
X = df.drop('Giá', axis=1).values
y = df['Giá'].values.reshape(-1, 1)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Thêm bias
X = np.hstack([np.ones((X.shape[0], 1)), X])

# 5. Chia tập train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 6. Linear Regression from scratch
def predict(theta, X):
    return np.dot(X, theta)

def compute_cost(theta, X, y):
    m = len(y)
    predictions = predict(theta, X)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = predict(theta, X)
        error = predictions - y
        gradient = (1 / m) * np.dot(X.T, error)
        theta -= learning_rate * gradient

        if i % 10 == 0:
            cost = compute_cost(theta, X, y)
            cost_history.append(cost)

    return theta, cost_history

# 7. Huấn luyện mô hình
theta = np.zeros((X_train.shape[1], 1))
theta_final, cost_history = gradient_descent(X_train, y_train, theta, learning_rate=0.01, iterations=1000)

# 8. Đánh giá
y_pred = predict(theta_final, X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 9. In hệ số
print("\nHệ số theta (weights):")
for i, coef in enumerate(theta_final):
    print(f"θ{i} = {coef[0]:.4f}")

# 10. Biểu đồ Cost Function
plt.plot(range(0, 1000, 10), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 11. Biểu đồ giá thực tế vs dự đoán
plt.scatter(range(len(y_test)), y_test, label='Thực tế', color='blue')
plt.scatter(range(len(y_test)), y_pred, label='Dự đoán', color='red')
plt.title('So sánh giá thực tế và dự đoán')
plt.xlabel('Chỉ mục mẫu')
plt.ylabel('Giá nhà')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Biểu đồ ảnh hưởng từng yếu tố đến giá (ghi nhãn trực tiếp trên trục x)
print("\nBiểu đồ ảnh hưởng của từng yếu tố đến giá:")

for col in df.columns:
    if col == 'Giá':
        continue

    plt.figure(figsize=(6, 4))

    if col in categorical_cols:
        sns.boxplot(x=df[col], y=df['Giá'], palette='Set2')
        plt.title(f'Ảnh hưởng của "{col}" đến Giá')
        plt.xlabel(f'{col}')
        plt.ylabel('Giá')
        plt.grid(True)

        # Ghi nhãn trực tiếp trên biểu đồ
        if col in custom_label_map:
            label_dict = custom_label_map[col]
            labels = [label_dict.get(val, val) for val in sorted(df[col].unique())]
            plt.xticks(ticks=sorted(df[col].unique()), labels=labels)

    else:
        sns.scatterplot(x=df[col], y=df['Giá'], color='dodgerblue', label='Dữ liệu mẫu')
        plt.title(f'{col} vs Giá')
        plt.xlabel(col)
        plt.ylabel('Giá')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

