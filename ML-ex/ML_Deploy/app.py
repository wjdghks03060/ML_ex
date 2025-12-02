import streamlit as st
import numpy as np
import pickle

@st.cache_resource
def load_model(): # 자원 캐싱 가능
    with open('models/iris_model_rfc.pkl','rb') as f:
        model = pickle.load(f)
    return model
model = load_model()

def get_image_path(prediction):
    if prediction == 0:
        return "static/setosa.jpg"
    elif prediction == 1:
        return "static/versicolor.jpg"
    else:
        return "static/virginica.png"

st.title("Iris 품종 예측")
st.write("꽃받침 길이, 너비, 너비를 입력하여 품종을 예측해보세요")
sepal_length = st.number_input("꽃받침 길이", min_value =0.0 , max_value =10.0 , value =5.0)
sepal_width = st.number_input("꽃받침 너비",min_value = 0.0, max_value =10.0, value =3.0)
petal_length = st.number_input("꽃잎 길이" , min_value = 0, max_value =10, value =4)
petal_width = st.number_input("꽃잎 너비",min_value = 0, max_value =10, value =1)

# 예측하기 버튼
if st.button("예측하기"):
    btn_clicked = True
else:
    btn_clicked = False
st.markdown('---------------------')
st.subheader("예측 결과 출력")

if btn_clicked:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # 예측 수행
    prediction = model.predict(input_data)
    predicted_class = prediction[0]
    # 예측된 클래스 이름
    class_name = ['Setosa', 'Versicolor', 'Virginica'][predicted_class]
    st.subheader(f"예측된 품종: {class_name}")

    # 예측된 품종에 해당하는 이미지 출력
    image_path = get_image_path(predicted_class)
    st.image(image_path, caption=class_name)

