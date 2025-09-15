"""
Week 08: AI 응용 프로젝트 - 뉴스 카테고리 분류 시스템
Streamlit을 이용한 웹 애플리케이션
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import re
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="뉴스 카테고리 분류기",
    page_icon="📰",
    layout="wide"
)

# 제목
st.title("📰 뉴스 카테고리 자동 분류 시스템")
st.markdown("---")

# 사이드바
st.sidebar.title("🔧 설정")
st.sidebar.markdown("### 프로젝트 정보")
st.sidebar.info("""
**Week 8 AI 응용 프로젝트**
- 뉴스 기사 자동 분류
- 머신러닝 기반 텍스트 분류
- 실시간 예측 시스템
""")

# 텍스트 전처리 함수
@st.cache_data
def preprocess_text(text):
    """텍스트 전처리"""
    # 특수문자 제거
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', '', text)
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

# 모델 학습 함수 (데모용)
@st.cache_resource
def train_demo_model():
    """데모용 뉴스 분류 모델 학습"""
    
    # 샘플 뉴스 데이터
    news_data = {
        'content': [
            # 정치
            "대통령이 오늘 국정현안에 대한 담화를 발표했습니다. 경제정책과 외교정책에 대한 향후 계획을 설명했습니다.",
            "국회에서 예산안 심의가 진행되고 있습니다. 여야 간 이견이 팽팽한 상황입니다.",
            "정부가 새로운 정책을 발표했습니다. 국민들의 관심이 집중되고 있습니다.",
            "선거 준비가 본격화되고 있습니다. 각 정당의 공약 발표가 이어지고 있습니다.",
            
            # 경제
            "주식시장이 상승세를 보이고 있습니다. 투자자들의 관심이 높아지고 있습니다.",
            "중앙은행이 기준금리를 조정했습니다. 시장에 미치는 영향이 주목됩니다.",
            "대기업의 실적 발표가 있었습니다. 매출과 순이익이 크게 증가했습니다.",
            "부동산 가격 변동에 대한 분석이 발표되었습니다. 전문가들의 의견이 분분합니다.",
            
            # 사회
            "교육정책 개편안이 발표되었습니다. 학생과 학부모들의 반응이 다양합니다.",
            "의료진 파업이 계속되고 있습니다. 환자들의 불편이 가중되고 있습니다.",
            "범죄 예방을 위한 새로운 시스템이 도입됩니다. 시민들의 안전 확보가 목표입니다.",
            "사회복지 제도 개선방안이 논의되고 있습니다. 취약계층 지원이 강화될 예정입니다.",
            
            # 스포츠
            "월드컵 예선전에서 우리나라가 승리했습니다. 선수들의 활약이 돋보였습니다.",
            "프로야구 시즌이 시작되었습니다. 팬들의 기대가 높아지고 있습니다.",
            "올림픽 준비가 한창입니다. 선수들의 훈련이 강화되고 있습니다.",
            "프로축구 리그에서 흥미진진한 경기가 펼쳐졌습니다. 관중들의 열띤 응원이 이어졌습니다.",
            
            # 기술
            "인공지능 기술이 빠르게 발전하고 있습니다. 다양한 분야에서 활용되고 있습니다.",
            "새로운 스마트폰이 출시되었습니다. 혁신적인 기능들이 탑재되었습니다.",
            "자율주행차 기술이 상용화 단계에 접어들었습니다. 교통 혁신이 기대됩니다.",
            "블록체인 기술의 활용 범위가 확대되고 있습니다. 금융권의 관심이 높습니다."
        ],
        'category': [
            '정치', '정치', '정치', '정치',
            '경제', '경제', '경제', '경제', 
            '사회', '사회', '사회', '사회',
            '스포츠', '스포츠', '스포츠', '스포츠',
            '기술', '기술', '기술', '기술'
        ]
    }
    
    df = pd.DataFrame(news_data)
    
    # 텍스트 전처리
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    # 파이프라인 생성
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words=None)),
        ('classifier', MultinomialNB(alpha=1.0))
    ])
    
    # 모델 학습
    pipeline.fit(df['processed_content'], df['category'])
    
    return pipeline, df

# 모델 로드
model, sample_data = train_demo_model()

# 메인 컨텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 뉴스 기사 분류")
    
    # 텍스트 입력
    news_text = st.text_area(
        "분류하고 싶은 뉴스 기사를 입력하세요:",
        height=200,
        placeholder="예: 대통령이 오늘 새로운 경제정책을 발표했습니다..."
    )
    
    # 예시 뉴스 선택
    st.subheader("📄 예시 뉴스 선택")
    selected_example = st.selectbox(
        "예시 뉴스를 선택하거나 직접 입력하세요:",
        ["직접 입력"] + list(sample_data['content'][:10])
    )
    
    if selected_example != "직접 입력":
        news_text = selected_example
        st.text_area("선택된 뉴스:", value=news_text, height=100, disabled=True)

with col2:
    st.header("📊 모델 정보")
    
    st.metric("훈련 데이터 수", len(sample_data))
    st.metric("카테고리 수", len(sample_data['category'].unique()))
    
    st.subheader("📝 지원 카테고리")
    categories = sample_data['category'].unique()
    for cat in categories:
        count = len(sample_data[sample_data['category'] == cat])
        st.write(f"• {cat}: {count}개")

# 분류 실행
if st.button("🚀 뉴스 분류하기", type="primary", use_container_width=True):
    if news_text.strip():
        with st.spinner("분류 중..."):
            # 전처리
            processed_text = preprocess_text(news_text)
            
            # 예측
            prediction = model.predict([processed_text])[0]
            probabilities = model.predict_proba([processed_text])[0]
            
            # 결과 표시
            st.success("분류 완료!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 예측 결과")
                st.metric(
                    label="카테고리",
                    value=prediction,
                    delta=f"확신도: {probabilities.max():.1%}"
                )
            
            with col2:
                st.subheader("📊 카테고리별 확률")
                prob_df = pd.DataFrame({
                    '카테고리': model.classes_,
                    '확률': probabilities
                }).sort_values('확률', ascending=False)
                
                st.dataframe(prob_df, use_container_width=True)
            
            # 확률 시각화
            st.subheader("📈 확률 분포")
            chart_data = prob_df.set_index('카테고리')
            st.bar_chart(chart_data)
            
            # 분석 결과 저장 (세션 상태)
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': news_text[:100] + "..." if len(news_text) > 100 else news_text,
                'prediction': prediction,
                'confidence': probabilities.max()
            })
    else:
        st.error("뉴스 기사 내용을 입력해주세요!")

# 분석 히스토리
if 'history' in st.session_state and st.session_state.history:
    st.header("📋 분석 히스토리")
    
    history_df = pd.DataFrame(st.session_state.history)
    
    # 최근 10개만 표시
    recent_history = history_df.tail(10)
    
    for i, row in recent_history.iterrows():
        with st.expander(f"{row['timestamp']} - {row['prediction']} ({row['confidence']:.1%})"):
            st.write(row['text'])
    
    # 히스토리 초기화 버튼
    if st.button("🗑️ 히스토리 초기화"):
        st.session_state.history = []
        st.rerun()

# 통계 대시보드
st.header("📊 분석 통계")

if 'history' in st.session_state and st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 분석 수", len(history_df))
    
    with col2:
        avg_confidence = history_df['confidence'].mean()
        st.metric("평균 확신도", f"{avg_confidence:.1%}")
    
    with col3:
        most_common = history_df['prediction'].mode().iloc[0] if len(history_df) > 0 else "없음"
        st.metric("가장 많은 카테고리", most_common)
    
    # 카테고리별 분포
    if len(history_df) > 0:
        category_counts = history_df['prediction'].value_counts()
        st.subheader("카테고리별 분석 현황")
        st.bar_chart(category_counts)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 <strong>AI 뉴스 분류 시스템</strong> | Week 8 멘토링 프로젝트</p>
    <p>머신러닝을 활용한 텍스트 분류 데모</p>
</div>
""", unsafe_allow_html=True)

# 사이드바 추가 정보
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 학습 내용")
st.sidebar.markdown("""
- 텍스트 전처리
- TF-IDF 벡터화
- 나이브 베이즈 분류
- Streamlit 웹앱 개발
""")

st.sidebar.markdown("### 🔗 다음 단계")
st.sidebar.markdown("""
1. 더 많은 데이터 수집
2. 딥러닝 모델 적용
3. 실시간 뉴스 크롤링
4. 모델 성능 개선
""")
