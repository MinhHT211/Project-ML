import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and preprocess the data
df = pd.read_csv("new_data.csv")
df.rename(columns={'Cở sở hạ tầng': 'Cơ sở hạ tầng'}, inplace=True)

# 2. Encode categorical columns with clear label maps
categorical_cols = ['Vị trí', 'Hướng', 'Tiện ích', 'Cơ sở hạ tầng', 'Xu hướng']
encoders = {}

# Custom English label mapping for visualization
custom_label_map = {
    'Vị trí': {0: 'Rural', 1: 'Urban'},
    'Hướng': {0: 'North', 1: 'East', 2: 'South', 3: 'West'},
    'Tiện ích': {0: 'Insufficient', 1: 'Sufficient'},
    'Cơ sở hạ tầng': {0: 'Poor', 1: 'Incomplete', 2: 'Complete'},
    'Xu hướng': {0: 'Falling', 1: 'Stable', 2: 'Rising'}
}

# Encode categorical columns and store LabelEncoders
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Split features and target
X = df.drop('Giá', axis=1).values
y = df['Giá'].values.reshape(-1, 1)

# 4. Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term
X = np.hstack([np.ones((X.shape[0], 1)), X])

# 5. Train-test split
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

# 7. Train the model
theta = np.zeros((X_train.shape[1], 1))
theta_final, cost_history = gradient_descent(X_train, y_train, theta, learning_rate=0.01, iterations=1000)

# 8. Evaluation
y_pred = predict(theta_final, X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 9. Print coefficients
print("\nTheta coefficients (weights):")
for i, coef in enumerate(theta_final):
    print(f"θ{i} = {coef[0]:.4f}")

# 10. Cost Function plot
plt.plot(range(0, 1000, 10), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.grid(True)
plt.show()

# 11. Actual vs Predicted Price
plt.scatter(range(len(y_test)), y_test, label='Actual', color='blue')
plt.scatter(range(len(y_test)), y_pred, label='Predicted', color='red')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Feature impact on price
print("\nEffect of each feature on house price:")

for col in df.columns:
    if col == 'Giá':
        continue

    plt.figure(figsize=(6, 4))

    if col in categorical_cols:
        sns.boxplot(x=df[col], y=df['Giá'], palette='Set2')
        plt.title(f'Effect of "{col}" on House Price')
        plt.xlabel(col)
        plt.ylabel('House Price')
        plt.grid(True)

        if col in custom_label_map:
            label_dict = custom_label_map[col]
            labels = [label_dict.get(val, val) for val in sorted(df[col].unique())]
            plt.xticks(ticks=sorted(df[col].unique()), labels=labels)

    else:
        sns.scatterplot(x=df[col], y=df['Giá'], color='dodgerblue', label='Sample Data')
        plt.title(f'{col} vs House Price')
        plt.xlabel(col)
        plt.ylabel('House Price')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
