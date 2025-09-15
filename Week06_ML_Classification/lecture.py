"""
Week 06: 머신러닝 심화 실습 - 분류 모델
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

print("=== 머신러닝 분류 모델 심화 실습 ===\n")

# 1. 분류 문제 소개
print("1. 분류(Classification)란?")
print("-" * 30)
print("🎯 분류: 데이터를 미리 정의된 카테고리로 분류하는 문제")
print("📝 예시: 스팸메일 분류, 이미지 인식, 질병 진단, 고객 분류")
print("🔢 이진 분류: Yes/No, 스팸/정상 (2개 클래스)")
print("🌈 다중 분류: 꽃의 종류, 뉴스 카테고리 (3개 이상 클래스)")
print()

# 2. Iris 데이터셋 로드
print("2. Iris 꽃 분류 데이터셋")
print("-" * 25)

# Iris 데이터 로드
iris = load_iris()
X = iris.data  # 특성: 꽃잎/꽃받침 길이와 너비
y = iris.target  # 타겟: 꽃의 종류 (0: setosa, 1: versicolor, 2: virginica)

# DataFrame 생성
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Categorical.from_codes(y, target_names)

print(f"📊 데이터 크기: {df.shape}")
print(f"🌸 꽃의 종류: {list(target_names)}")
print(f"📏 특성: {list(feature_names)}")
print("\n데이터 미리보기:")
print(df.head())
print("\n클래스별 데이터 개수:")
print(df['species'].value_counts())
print()

# 3. 데이터 탐색 및 시각화
print("3. 데이터 탐색 및 시각화")
print("-" * 25)

plt.figure(figsize=(15, 10))

# 특성별 분포 (종류별)
for i, feature in enumerate(feature_names):
    plt.subplot(2, 3, i+1)
    for species in target_names:
        data = df[df['species'] == species][feature]
        plt.hist(data, alpha=0.7, label=species, bins=15)
    plt.xlabel(feature)
    plt.ylabel('빈도')
    plt.title(f'{feature} 분포')
    plt.legend()

# 산점도 매트릭스
plt.subplot(2, 3, 5)
colors = ['red', 'green', 'blue']
for i, species in enumerate(target_names):
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
               species_data['petal length (cm)'], 
               c=colors[i], label=species, alpha=0.7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Sepal vs Petal Length')
plt.legend()

# 상관관계 히트맵
plt.subplot(2, 3, 6)
correlation = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('특성간 상관관계')

plt.tight_layout()
plt.show()

# 4. 데이터 전처리 및 분할
print("4. 데이터 전처리 및 분할")
print("-" * 25)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"훈련 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")

# 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ 데이터 정규화 완료")
print()

# 5. 다양한 분류 모델 학습 및 비교
print("5. 다양한 분류 모델 비교")
print("-" * 25)

# 모델들 정의
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("🤖 모델 학습 및 평가:")
print("-" * 40)

for name, model in models.items():
    # 모델 학습
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    
    # 교차검증 점수
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"📊 {name}:")
    print(f"  테스트 정확도: {accuracy:.3f}")
    print(f"  교차검증 평균: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print()

# 6. 최고 성능 모델 상세 분석
print("6. 최고 성능 모델 상세 분석")
print("-" * 30)

# 가장 좋은 모델 찾기
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"🏆 최고 성능 모델: {best_model_name}")
print(f"정확도: {results[best_model_name]['accuracy']:.3f}")
print()

# 분류 리포트
print("📋 상세 분류 리포트:")
print(classification_report(y_test, best_predictions, target_names=target_names))

# 혼동 행렬
print("🔍 혼동 행렬 (Confusion Matrix):")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'{best_model_name}\n혼동 행렬')
plt.xlabel('예측 클래스')
plt.ylabel('실제 클래스')

# 모델별 성능 비교
plt.subplot(1, 2, 2)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_means = [results[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='테스트 정확도', alpha=0.8)
plt.bar(x + width/2, cv_means, width, label='교차검증 평균', alpha=0.8)

plt.xlabel('모델')
plt.ylabel('정확도')
plt.title('모델별 성능 비교')
plt.xticks(x, [name.split()[0] for name in model_names], rotation=45)
plt.legend()
plt.ylim(0.8, 1.0)

# 정확도 값 표시
for i, (acc, cv) in enumerate(zip(accuracies, cv_means)):
    plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    plt.text(i + width/2, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 7. 특성 중요도 분석 (Random Forest)
print("7. 특성 중요도 분석")
print("-" * 20)

if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = rf_model.feature_importances_
    
    # 특성 중요도 데이터프레임
    importance_df = pd.DataFrame({
        '특성': feature_names,
        '중요도': feature_importance
    }).sort_values('중요도', ascending=False)
    
    print("🔍 특성 중요도 순위:")
    for i, row in importance_df.iterrows():
        print(f"  {row['특성']}: {row['중요도']:.3f}")
    
    # 특성 중요도 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='중요도', y='특성', palette='viridis')
    plt.title('Random Forest 특성 중요도')
    plt.xlabel('중요도')
    
    # 값 표시
    for i, v in enumerate(importance_df['중요도']):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

print()

# 8. 새로운 데이터 예측 실습
print("8. 새로운 데이터 예측 실습")
print("-" * 25)

# 새로운 꽃 데이터 (임의 생성)
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # setosa와 유사
    [6.2, 2.8, 4.8, 1.8],  # versicolor와 유사
    [7.3, 2.9, 6.3, 1.8]   # virginica와 유사
])

# 정규화
new_flowers_scaled = scaler.transform(new_flowers)

# 예측
predictions = best_model.predict(new_flowers_scaled)
probabilities = best_model.predict_proba(new_flowers_scaled)

print("🔮 새로운 꽃 분류 예측:")
for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
    predicted_species = target_names[pred]
    confidence = prob.max()
    
    print(f"\n꽃 {i+1}: {flower}")
    print(f"  예측 종류: {predicted_species}")
    print(f"  확신도: {confidence:.2f}")
    print(f"  각 클래스 확률:")
    for j, (species, p) in enumerate(zip(target_names, prob)):
        print(f"    {species}: {p:.3f}")

# 9. 모델 성능 개선 팁
print("\n" + "="*50)
print("9. 분류 모델 성능 개선 팁")
print("="*50)
print("💡 데이터 관련:")
print("  - 더 많은 데이터 수집")
print("  - 데이터 품질 개선 (이상치, 결측치 처리)")
print("  - 특성 엔지니어링 (새로운 특성 생성)")
print()
print("🔧 모델 관련:")
print("  - 하이퍼파라미터 튜닝")
print("  - 앙상블 기법 활용")
print("  - 교차검증으로 안정성 확인")
print()
print("⚖️ 평가 관련:")
print("  - 적절한 평가 지표 선택")
print("  - 클래스 불균형 고려")
print("  - 비즈니스 목표와 연결")

print("\n=== 분류 모델 심화 학습 완료! ===")
print("🎉 다양한 분류 알고리즘을 비교해보았습니다!")
print("💡 다음 주차에서는 텍스트 데이터 처리를 배워보겠습니다.")
