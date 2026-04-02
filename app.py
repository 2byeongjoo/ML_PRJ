# 1. 라이브러리 불러오기
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 산점도 라벨 깨짐 방지 → 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 2. 모델과 스케일러 pkl 불러오기
# model.pkl 불러옴
with open('model.pkl','rb') as f:
    model = pickle.load(f)
    
# 스케일러.pkl 불러옴
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# 3. 페이지 제목 및 설명
st.title('수면 품질 예측 시스템')
st.write('🌙 오늘 밤, 당신의 수면은 몇 점일까요?')
st.write('생활 패턴을 입력하면 AI가 수면 품질을 예측해드립니다.')

# 4. 사이드바 또는 입력 폼 구성
# (수면시간, 나이, 카페인 섭취량, 각성횟수, 운동빈도)
sleep_duration  = st.slider('수면시간', 1.0, 12.0, 7.0)                          # '라벨', 최소값, 최대값, 기본값
age = st.number_input('나이', min_value=1, max_value=100, value=25)              # '라벨', 최소값, 최대값, 기본값
caffeine = st.number_input('카페인', min_value=0, max_value=200, value=50)        # '라벨', 최소값, 최대값, 기본값
awakenings = st.number_input('각성횟수', min_value=0, max_value=10, value=1)       # '라벨', 최소값, 최대값, 기본값
exercise = st.number_input('운동빈도(회/주)', min_value=0, max_value=7, value=3)    # '라벨', 최소값, 최대값, 기반값

# 5. 예측 버튼 클릭 시
if st.button('예측하기'): # 클릭 시 True값이 할당됨.
    # 입력값 numpy 배열로 변환
    input_data = np.array([[sleep_duration, age, caffeine, awakenings, exercise]])
    
    # - 스케일러로 변환
    input_scaled = scaler.transform(input_data)
    
    # - 모델로 예측
    prediction = model.predict(input_scaled)
    
    # - 결과 출력 (점수 + 등급 + 피드백)
    st.write(f"예측 수면 품질 점수: {prediction[0]:.1f}점")


    # 6. 예측 이력 MariaDB 저장 (선택)

    # 7. 데이터 시각화
    # 데이터 불러오기
    df = pd.read_csv('dataset/Sleep_Efficiency.csv')

    # - Heatmap
    st.subheader('피처 간 상관관계 히트맵')
    st.subheader('피처 간 상관관계 히트맵')
    fig, ax = plt.subplots()

    # ❌ 기존 코드 (Sleep efficiency 포함)
    # sns.heatmap(df.corr(numeric_only=True), ax=ax, annot=True, fmt='.1f', cmap='coolwarm')

    # ✅ 수정 코드 (Sleep efficiency 제외)
    heatmap_cols = ['Age', 'Sleep duration', 'Caffeine consumption', 
                    'Awakenings', 'Exercise frequency', 'Quality_of_Sleep']
    # df에 Quality_of_Sleep 컬럼이 없으면 추가
    if 'Quality_of_Sleep' not in df.columns:
        df['Quality_of_Sleep'] = df['Sleep efficiency'] * 10

    sns.heatmap(df[heatmap_cols].corr(), 
                ax=ax, 
                annot=True, 
                fmt='.2f',  # .1f → .2f로 변경
                cmap='coolwarm')
    st.pyplot(fig)

    # - 산점도용 데이터 준비
    from sklearn.model_selection import train_test_split

    df2 = pd.read_csv('dataset/Sleep_Efficiency.csv')
    df2 = df2[['Age', 'Sleep duration', 'Caffeine consumption',
            'Awakenings', 'Exercise frequency', 'Sleep efficiency']]
    df2 = df2.fillna(df2.mean())
    df2['Quality_of_Sleep'] = df2['Sleep efficiency'] * 10
    X2 = df2.iloc[:, :-2]
    y2 = df2.iloc[:, -1]
    _, X_test2, _, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
    X_test2_scaled = scaler.transform(X_test2)
    y_pred2 = model.predict(X_test2_scaled)

    # - 산점도
    st.subheader('실제값 vs 예측값 산점도')
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test2, y_pred2, alpha=0.6)
    ax2.plot([y_test2.min(), y_test2.max()],
            [y_test2.min(), y_test2.max()], 'r--', linewidth=2)
    ax2.set_xlabel('실제 수면 품질')
    ax2.set_ylabel('예측 수면 품질')
    st.pyplot(fig2)