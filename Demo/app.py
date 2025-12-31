import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán mức độ béo phì",
    layout="centered"
)

# Load mô hình đã huấn luyện
@st.cache_resource
def load_model():
    try:
        model = joblib.load('Model/obesity_pipeline.pkl')
        return model
    except FileNotFoundError:
        st.error("Không tìm thấy file 'Model/obesity_pipeline.pkl'. Vui lòng huấn luyện và lưu pipeline từ notebook trước.")
        st.stop()

pipeline = load_model()

# TIÊU ĐỀ & GIỚI THIỆU

st.title("Dự đoán mức độ béo phì 🏋️‍♂️")
st.markdown("""
Ứng dụng này sử dụng mô hình **Random Forest** để dự đoán mức độ béo phì dựa trên thông tin cá nhân và thói quen sống.
""")

# FORM NHẬP LIỆU

with st.form("user_input"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Giới tính", ["Male", "Female"])
        age = st.number_input("Tuổi", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        height = st.number_input("Chiều cao (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Cân nặng (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

    with col2:
        family_history = st.selectbox("Tiền sử gia đình thừa cân?", ["Yes", "No"])
        favc = st.selectbox("Thường xuyên ăn đồ ăn nhiều năng lượng cao?", ["Yes", "No"])
        fcvc = st.slider("Tần suất ăn rau củ (1–3)", 1.0, 3.0, 2.0, step=0.1)
        ncp = st.slider("Số bữa chính mỗi ngày (1–4)", 1.0, 4.0, 3.0, step=0.1)
        caec = st.selectbox("Ăn vặt giữa các bữa", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Hút thuốc?", ["Yes", "No"])
        ch2o = st.slider("Lượng nước uống mỗi ngày (1–3)", 1.0, 3.0, 2.0, step=0.1)

    col3, col4 = st.columns(2)

    with col3:
        scc = st.selectbox("Có theo dõi lượng calo không?", ["Yes", "No"])
        faf = st.slider("Tần suất hoạt động thể chất (0–3)", 0.0, 3.0, 1.0, step=0.1)
        tue = st.slider("Thời gian dùng thiết bị điện tử (0–2)", 0.0, 2.0, 1.0, step=0.1)

    with col4:
        calc = st.selectbox("Tần suất uống rượu", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Phương tiện di chuyển chính", 
                              ["Public_Transportation", "Automobile", "Motorbike", "Bike", "Walking"])

    # Nút submit
    submitted = st.form_submit_button("Dự đoán", type="primary", use_container_width=True)

# XỬ LÝ KHI NHẤN DỰ ĐOÁN

if submitted:
    # Chuẩn bị dữ liệu đầu vào đúng định dạng
    input_data = {
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [1 if family_history == "Yes" else 0],
        "FAVC": [1 if favc == "Yes" else 0],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": [caec],
        "SMOKE": [1 if smoke == "Yes" else 0],
        "CH2O": [ch2o],
        "SCC": [1 if scc == "Yes" else 0],
        "FAF": [faf],
        "TUE": [tue],
        "CALC": [calc],
        "MTRANS": [mtrans]
    }

    df_input = pd.DataFrame(input_data)

    # Dự đoán
    with st.spinner("Đang dự đoán..."):
        prediction = pipeline.predict(df_input)[0]

    # Hiển thị kết quả
    st.success(f"Kết quả dự đoán: **{prediction}**")

    # Hiển thị giải thích ngắn gọn
    explanations = {
        "Insufficient_Weight": "Thiếu cân",
        "Normal_Weight": "Cân nặng bình thường",
        "Overweight_Level_I": "Thừa cân cấp 1",
        "Overweight_Level_II": "Thừa cân cấp 2",
        "Obesity_Type_I": "Béo phì loại I",
        "Obesity_Type_II": "Béo phì loại II",
        "Obesity_Type_III": "Béo phì loại III (nặng)"
    }

    st.markdown(f"**Giải thích:** {explanations.get(prediction, 'Không xác định')}")

    # Gợi ý (tùy chọn)
    if "Obesity" in prediction or "Overweight" in prediction:
        st.warning("Kết quả cho thấy có nguy cơ thừa cân / béo phì. Nên tham khảo ý kiến bác sĩ hoặc chuyên gia dinh dưỡng.")
    elif prediction == "Insufficient_Weight":
        st.info("Cân nặng đang ở mức thấp. Nên chú ý bổ sung dinh dưỡng hợp lý.")
    else:
        st.info("Cân nặng đang ở mức bình thường. Duy trì lối sống lành mạnh nhé!")