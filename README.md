# Dự đoán mức độ béo phì bằng các thuật toán Machine Learning
## Bài toán
Bài toán nhằm dự đoán mức độ béo phì của một cá nhân dựa trên các thông tin về nhân khẩu học, thói quen ăn uống, hoạt động thể chất và phương tiện di chuyển. Mục tiêu là xây dựng mô hình phân loại giúp nhận diện sớm các mức độ thừa cân / béo phì, hỗ trợ can thiệp sức khỏe kịp thời.

Bài toán được mô hình hóa dưới dạng phân loại đa lớp với 7 nhãn mục tiêu:

- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

## Dataset

Nguồn dữ liệu: Kaggle
```text
https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset?resource=download
```
Dữ liệu bao gồm thông tin tại thời điểm khảo sát:

- Thông tin nhân khẩu học
- Thói quen ăn uống
- Mức độ hoạt động thể chất
- Tần suất sử dụng thiết bị điện tử
- Phương tiện di chuyển chủ yếu
- Tiền sử gia đình về thừa cân / béo phì

## Thuộc tính dữ liệu

-Thông tin nhân khẩu học
  + Gender: Giới tính
  + Age: Tuổi
  + Height: Chiều cao (m)
  + Weight: Cân nặng (kg)

- Yếu tố gia đình & thói quen ăn uống
  + family_history_with_overweight: Tiền sử gia đình thừa cân (0/1)
  + FAVC: Thường xuyên tiêu thụ thực phẩm năng lượng cao (0/1)
  FCVC: Tần suất tiêu thụ rau củ
  + NCP: Số bữa chính mỗi ngày
  + CAEC: Tiêu thụ thức ăn giữa các bữa
  + SMOKE: Hút thuốc lá (0/1)
  + CH2O: Lượng nước uống hàng ngày
  + SCC: Theo dõi lượng calo tiêu thụ (0/1)
  + CALC: Tần suất tiêu thụ rượu

- Hoạt động thể chất & lối sống
  + FAF: Tần suất hoạt động thể chất
  + TUE: Thời gian sử dụng thiết bị điện tử
  + MTRANS: Phương tiện di chuyển chính

- Nhãn mục tiêu
0be1dad: Mức độ béo phì (7 lớp như liệt kê ở trên)

## Pipeline
Dataset → EDA → Preprocessing → Train → Evaluate → Inference

## Các bước chính đã triển khai
- Phân tách đặc trưng số và phân loại
- Chuẩn hóa dữ liệu số bằng StandardScaler
- Mã hóa one-hot cho các biến phân loại bằng OneHotEncoder
- Sử dụng ColumnTransformer để xử lý đồng thời các loại đặc trưng
- Xây dựng Pipeline tích hợp tiền xử lý + mô hình
- Huấn luyện và đánh giá mô hình
- Dự đoán trên dữ liệu mới

## Mô hình đã triển khai

- Random Forest
- Logistic Regression

## Kết quả
Phân loại đa lớp – Random Forest (mô hình tốt nhất)
| Metric              | Train  |  Test  | Ghi chú                  |
|---------------------|-------:|-------:|-------------------------:|
| Accuracy            | 0.9187 | 0.8964 | Chênh lệch nhỏ           |
| Macro avg F1        |   -    | ~0.89  | Hiệu suất trung bình tốt |
| Weighted avg F1     |   -    | ~0.90  | Tốt khi xét tỷ lệ mẫu    |

##Điểm nổi bật:

- Obesity_Type_III: Precision & Recall ≈ 1.00 (dễ phân biệt nhất)
- Insufficient_Weight và Obesity_Type_II: F1 > 0.90
- Overweight_Level_I và Overweight_Level_II: Lớp khó nhất (F1 ~0.75–0.82)

So sánh với các mô hình khác
Random Forest vượt trội hơn rõ rệt so với Logistic Regression và SVC về khả năng phân biệt các lớp thừa cân / béo phì mức độ trung bình.
Cách chạy
Chạy trên Jupyter Notebook (khuyến nghị)

Mở file Machine_Learning.ipynb
Chạy tuần tự các cell từ trên xuống dưới
Đảm bảo đã cài đặt các thư viện: Bashpip install numpy pandas seaborn scikit-learn matplotlib

Thử dự đoán trên dữ liệu mới
Sau khi huấn luyện pipeline (pipeline_rf), có thể dự đoán như sau:
Pythonsample = {
    "Gender": ["Male"],
    "Age": [25],
    "Height": [1.70],
    "Weight": [80],
    "family_history_with_overweight": [0],
    "FAVC": [1],
    # ... điền đầy đủ các trường còn lại
}

pred = pipeline_rf.predict(pd.DataFrame(sample))
print("Dự đoán:", pred)


Tác giả
Họ và tên: Hoàng Công Hoàn
Mã sinh viên: 10123137
Lớp: 12423TN
Dự án tập trung vào việc áp dụng Random Forest kết hợp pipeline tiền xử lý để đạt hiệu suất tốt trên bài toán phân loại mức độ béo phì đa lớp.
