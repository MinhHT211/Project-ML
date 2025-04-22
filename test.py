import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Đọc dữ liệu từ file CSV
df = pd.read_csv('new_data.csv')

# Xác định các cột đặc trưng (features) và cột mục tiêu (target)
X = df[['Vị trí', 'Hướng', 'Tiện ích', 'Cở sở hạ tầng', 'Xu hướng', 'Diện tích']]
y = df['Giá']

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cấu hình OneHotEncoder với handle_unknown='ignore' để xử lý các giá trị chưa thấy trong dữ liệu kiểm tra
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Vị trí', 'Hướng', 'Tiện ích', 'Cở sở hạ tầng', 'Xu hướng']),
        ('num', 'passthrough', ['Diện tích'])  # Không thay đổi cột 'Diện tích'
    ]
)

# Tạo pipeline kết hợp tiền xử lý và mô hình Linear Regression
pipeline = Pipeline(steps=[
    ('preprocessor', column_transformer),
    ('model', LinearRegression())
])

# Huấn luyện mô hình với pipeline
pipeline.fit(X_train, y_train)

# Lưu mô hình đã huấn luyện vào file pickle
pickle.dump(pipeline, open('LandPricePredictionModel.pkl', 'wb'))

# Đánh giá mô hình trên dữ liệu kiểm tra
test_score = pipeline.score(X_test, y_test)
print(f"R^2 trên dữ liệu kiểm tra: {test_score:.2f}")
