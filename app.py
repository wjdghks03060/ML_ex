import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# -----------------------------
# 모델 로드 함수
# -----------------------------
@st.cache_resource
def load_model():
    current_dir = Path.cwd()  # Streamlit Cloud 환경에서 현재 작업 디렉토리
    model_path = current_dir / "models" / "iris_model_rfc.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# -----------------------------
# 예측 결과에 맞는 이미지 경로 반환
# -----------------------------
def get_image_path(prediction):
    current_dir = Path.cwd()
    images = {
        0: current_dir / "static" / "setosa.jpg",
        1: current_dir / "static" / "versicolor.jpg",
        2: current_dir / "static" / "virginica.png"
    }
    return images[prediction]

# -----------------------------
# 모델 로드
# -----------------------------
model = load_model()

# -----------------------------
# Streamlit 앱 UI
# -----------------------------
st.title("Iris 품종 예측")
st.write("꽃받침 길이, 너비, 꽃잎 길이, 너비를 입력하여 품종을 예측해보세요")

sepal_length = st.number_input("꽃받침 길이", min_value=0.0, max_value=10.0, value=5.0)
sepal_width  = st.number_input("꽃받침 너비", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("꽃잎 길이", min_value=0.0, max_value=10.0, value=4.0)
petal_width  = st.number_input("꽃잎 너비", min_value=0.0, max_value=10.0, value=1.0)

# 예측 버튼
btn_clicked = st.button("예측하기")

st.markdown('---------------------')
st.subheader("예측 결과 출력")

# 예측 수행
if btn_clicked:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = prediction[0]

    # 클래스 이름
    class_name = ['Setosa', 'Versicolor', 'Virginica'][predicted_class]
    st.subheader(f"예측된 품종: {class_name}")

    # 이미지 출력
    image_path = get_image_path(predicted_class)
    st.image(str(image_path), caption=class_name)
