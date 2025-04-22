import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

# Load mô hình (pipeline) đã huấn luyện
model = pickle.load(open('LandPricePredictionModel.pkl', 'rb'))

app = Flask(__name__)

# Các cột đầu vào, đúng với lúc huấn luyện
columns = ['Vị trí', 'Hướng', 'Tiện ích', 'Cở sở hạ tầng', 'Xu hướng', 'Diện tích']

@app.route('/')
def index():
    return render_template('index.html',
                           prediction_text='',
                           location='',
                           direction='',
                           amenities='',
                           infrastructure='',
                           trend='',
                           area='')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy dữ liệu từ form
        location = request.form['location']
        direction = request.form['direction']
        amenities = request.form['amenities']
        infrastructure = request.form['infrastructure']
        trend = request.form['trend']
        area = float(request.form['area'])

        # Tạo dataframe đúng định dạng cho mô hình
        input_data = pd.DataFrame([[location, direction, amenities, infrastructure, trend, area]],
                                  columns=columns)

        # Dự đoán giá đất
        predicted_price = model.predict(input_data)[0]

        # Trả kết quả + giữ lại dữ liệu đã nhập
        return render_template('index.html',
                               prediction_text=f'Land price prediction: {predicted_price:.2f} Billion VND',
                               location=location,
                               direction=direction,
                               amenities=amenities,
                               infrastructure=infrastructure,
                               trend=trend,
                               area=area)

if __name__ == '__main__':
    app.run(debug=True)
