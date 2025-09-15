"""
Week 07: 텍스트 데이터 처리 기초 - 자연어처리 & 감성분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

print("=== 텍스트 데이터 처리 기초 ===\n")

# 1. 자연어처리 소개
print("1. 자연어처리(NLP)란?")
print("-" * 25)
print("📝 자연어처리: 컴퓨터가 인간의 언어를 이해하고 처리하는 기술")
print("🔍 주요 응용:")
print("  - 감성 분석 (리뷰, 소셜미디어)")
print("  - 기계 번역 (구글 번역)")
print("  - 챗봇 및 가상 비서")
print("  - 문서 분류 및 요약")
print("  - 스팸 메일 필터링")
print()

# 2. 텍스트 전처리 기초
print("2. 텍스트 전처리 기초")
print("-" * 20)

# 예시 문서들
documents = [
    "이 영화는 정말 재미있었습니다! 강력 추천합니다.",
    "너무 지루한 영화였어요. 시간 아까웠습니다.",
    "배우들의 연기가 훌륭했고 스토리도 감동적이었습니다.",
    "별로였어요. 돈이 아까운 영화입니다.",
    "최고의 영화! 다시 보고 싶어요."
]

print("📄 원본 텍스트:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc}")
print()

# 기본 전처리 함수들
def clean_text(text):
    """기본 텍스트 정제"""
    # 특수문자 제거 (한글, 영어, 숫자, 공백만 남김)
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', '', text)
    # 연속 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

def simple_tokenize(text):
    """간단한 토큰화 (공백 기준)"""
    return text.split()

print("🧹 전처리 후:")
cleaned_docs = [clean_text(doc) for doc in documents]
for i, doc in enumerate(cleaned_docs, 1):
    print(f"{i}. {doc}")
    print(f"   토큰: {simple_tokenize(doc)}")
print()

# 3. 감성 분석용 데이터 준비
print("3. 감성 분석용 데이터 준비")
print("-" * 25)

# 영화 리뷰 예시 데이터 (더 많은 데이터)
movie_reviews = [
    # 긍정 리뷰
    "이 영화는 정말 훌륭했습니다 감동적이고 재미있었어요",
    "최고의 영화입니다 배우들의 연기가 뛰어났어요",
    "강력 추천합니다 스토리가 매우 흥미로웠습니다",
    "정말 재미있게 봤습니다 다시 보고 싶어요",
    "감동적인 영화였어요 눈물이 났습니다",
    "훌륭한 작품이네요 시간가는 줄 몰랐어요",
    "최고예요 정말 좋았습니다",
    "멋진 영화였습니다 추천해요",
    "정말 즐거웠어요 좋은 영화입니다",
    "감동받았습니다 훌륭한 스토리예요",
    
    # 부정 리뷰  
    "너무 지루한 영화였어요 시간 아까웠습니다",
    "별로였습니다 돈이 아까워요",
    "재미없어요 끝까지 보기 힘들었습니다",
    "실망스러운 영화였습니다 기대가 컸는데",
    "스토리가 엉성했어요 아쉬운 작품입니다",
    "지루하고 재미없었어요 추천하지 않습니다",
    "최악의 영화입니다 시간낭비였어요",
    "너무 아쉬운 작품이에요 기대 이하였습니다",
    "별로예요 돈아까운 영화입니다",
    "재미없고 지루했어요 실망했습니다"
]

# 라벨 생성 (1: 긍정, 0: 부정)
labels = [1] * 10 + [0] * 10  # 긍정 10개, 부정 10개

# DataFrame 생성
df = pd.DataFrame({
    'review': movie_reviews,
    'sentiment': labels
})

print(f"📊 데이터 크기: {df.shape}")
print(f"감성별 분포:")
print(df['sentiment'].value_counts())
print("\n데이터 예시:")
print(df.head())
print()

# 4. 단어 빈도 분석
print("4. 단어 빈도 분석")
print("-" * 15)

# 모든 텍스트 합치기
all_text = ' '.join(df['review'])
words = simple_tokenize(all_text)

# 단어 빈도 계산
word_freq = Counter(words)
most_common = word_freq.most_common(10)

print("🔤 가장 자주 나오는 단어들:")
for word, freq in most_common:
    print(f"  {word}: {freq}번")
print()

# 워드클라우드 생성 (한글 폰트 문제로 간단히 시각화)
plt.figure(figsize=(12, 5))

# 단어 빈도 막대그래프
plt.subplot(1, 2, 1)
words_list, freqs = zip(*most_common)
plt.barh(words_list, freqs)
plt.title('단어 빈도')
plt.xlabel('빈도')

# 감성별 단어 길이 분포
plt.subplot(1, 2, 2)
df['word_count'] = df['review'].apply(lambda x: len(simple_tokenize(x)))

for sentiment in [0, 1]:
    sentiment_data = df[df['sentiment'] == sentiment]['word_count']
    label = '부정' if sentiment == 0 else '긍정'
    plt.hist(sentiment_data, alpha=0.7, label=label, bins=10)

plt.xlabel('단어 개수')
plt.ylabel('빈도')
plt.title('감성별 리뷰 길이 분포')
plt.legend()

plt.tight_layout()
plt.show()

# 5. 텍스트 벡터화
print("5. 텍스트 벡터화")
print("-" * 15)

# 데이터 분할
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"훈련 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# 1) Bag of Words (BoW)
print("\n📊 Bag of Words 벡터화:")
bow_vectorizer = CountVectorizer(max_features=50, stop_words=None)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

print(f"BoW 특성 수: {X_train_bow.shape[1]}")
print(f"특성 예시: {list(bow_vectorizer.get_feature_names_out())[:10]}")

# 2) TF-IDF
print("\n📈 TF-IDF 벡터화:")
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words=None)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF 특성 수: {X_train_tfidf.shape[1]}")
print()

# 6. 감성 분석 모델 학습
print("6. 감성 분석 모델 학습")
print("-" * 20)

# 모델들 정의
models = {
    'Naive Bayes (BoW)': (MultinomialNB(), X_train_bow, X_test_bow),
    'Naive Bayes (TF-IDF)': (MultinomialNB(), X_train_tfidf, X_test_tfidf),
    'Logistic Regression (BoW)': (LogisticRegression(random_state=42), X_train_bow, X_test_bow),
    'Logistic Regression (TF-IDF)': (LogisticRegression(random_state=42), X_train_tfidf, X_test_tfidf)
}

results = {}

print("🤖 모델 학습 및 평가:")
for name, (model, X_tr, X_te) in models.items():
    # 모델 학습
    model.fit(X_tr, y_train)
    
    # 예측
    y_pred = model.predict(X_te)
    
    # 정확도 계산
    accuracy = (y_pred == y_test).mean()
    results[name] = accuracy
    
    print(f"📊 {name}: {accuracy:.3f}")

print()

# 최고 성능 모델 선택
best_model_name = max(results, key=results.get)
best_model, best_X_train, best_X_test = models[best_model_name]

print(f"🏆 최고 성능 모델: {best_model_name}")
print(f"정확도: {results[best_model_name]:.3f}")

# 상세 분류 리포트
y_pred_best = best_model.predict(best_X_test)
print("\n📋 상세 분류 리포트:")
print(classification_report(y_test, y_pred_best, target_names=['부정', '긍정']))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['부정', '긍정'], yticklabels=['부정', '긍정'])
plt.title(f'{best_model_name}\n혼동 행렬')
plt.xlabel('예측')
plt.ylabel('실제')
plt.show()

print()

# 7. 새로운 리뷰 감성 예측
print("7. 새로운 리뷰 감성 예측")
print("-" * 20)

# 새로운 리뷰들
new_reviews = [
    "정말 최고의 영화였습니다 감동받았어요",
    "너무 지루하고 재미없었습니다",
    "배우들의 연기가 훌륭했어요 추천합니다",
    "돈이 아까운 영화였어요 별로입니다"
]

# 최고 성능 모델의 벡터화 방법 확인
if 'TF-IDF' in best_model_name:
    vectorizer = tfidf_vectorizer
else:
    vectorizer = bow_vectorizer

# 새로운 리뷰 벡터화
new_reviews_vectorized = vectorizer.transform(new_reviews)

# 예측
predictions = best_model.predict(new_reviews_vectorized)
probabilities = best_model.predict_proba(new_reviews_vectorized)

print("🔮 새로운 리뷰 감성 예측:")
for i, (review, pred, prob) in enumerate(zip(new_reviews, predictions, probabilities)):
    sentiment = '긍정' if pred == 1 else '부정'
    confidence = prob.max()
    
    print(f"\n리뷰 {i+1}: {review}")
    print(f"  예측 감성: {sentiment}")
    print(f"  확신도: {confidence:.3f}")
    print(f"  확률 분포: 부정 {prob[0]:.3f}, 긍정 {prob[1]:.3f}")

# 8. 모델 해석 (특성 중요도)
print(f"\n8. 모델 해석 - {best_model_name}")
print("-" * 30)

if hasattr(best_model, 'coef_'):
    # 로지스틱 회귀의 경우 계수 확인
    feature_names = vectorizer.get_feature_names_out()
    coefficients = best_model.coef_[0]
    
    # 긍정적 단어들 (계수가 높은 단어)
    positive_indices = coefficients.argsort()[-10:][::-1]
    print("😊 긍정 감성에 기여하는 단어들:")
    for idx in positive_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")
    
    print()
    
    # 부정적 단어들 (계수가 낮은 단어)
    negative_indices = coefficients.argsort()[:10]
    print("😞 부정 감성에 기여하는 단어들:")
    for idx in negative_indices:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")

print("\n" + "="*50)
print("9. 텍스트 분석 성능 개선 방법")
print("="*50)
print("📝 데이터 전처리:")
print("  - 불용어 제거 (조사, 어미 등)")
print("  - 형태소 분석 (명사, 형용사 추출)")
print("  - 맞춤법 교정")
print("  - 동의어 처리")
print()
print("🔧 특성 엔지니어링:")
print("  - N-gram 활용 (2-gram, 3-gram)")
print("  - 단어 임베딩 (Word2Vec, FastText)")
print("  - 감정 사전 활용")
print()
print("🤖 모델 개선:")
print("  - 딥러닝 모델 (LSTM, BERT)")
print("  - 앙상블 기법")
print("  - 하이퍼파라미터 튜닝")

print("\n=== 텍스트 데이터 처리 기초 학습 완료! ===")
print("🎉 감성 분석 모델을 성공적으로 구축했습니다!")
print("💡 다음 주차에서는 실제 AI 응용 프로젝트를 진행해보겠습니다.")
