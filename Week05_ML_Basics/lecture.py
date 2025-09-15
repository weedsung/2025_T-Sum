"""
Week 05: 머신러닝 기본 개념 - 회귀 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

print("=== 머신러닝 기본 개념 - 회귀 분석 ===\n")

# 1. 머신러닝 개념 소개
print("1. 머신러닝이란?")
print("-" * 30)
print("🤖 머신러닝: 컴퓨터가 데이터로부터 패턴을 학습하여 예측하는 기술")
print("📊 지도학습: 정답(타겟)이 있는 데이터로 학습")
print("🔍 비지도학습: 정답 없이 데이터의 패턴을 찾는 학습")
print("🎯 강화학습: 보상을 통해 최적의 행동을 학습")
print()

# 2. 회귀 분석 예제 - 공부시간과 성적
print("2. 회귀 분석 실습: 공부시간 → 성적 예측")
print("-" * 40)

# 가상의 학습 데이터 생성
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 100)  # 1~10시간
# 실제 관계: 점수 = 50 + 4*공부시간 + 노이즈
scores = 50 + 4 * study_hours + np.random.normal(0, 5, 100)
scores = np.clip(scores, 0, 100)  # 0~100점 범위로 제한

# DataFrame 생성
df = pd.DataFrame({
    '공부시간': study_hours,
    '성적': scores
})

print("📊 데이터 정보:")
print(f"데이터 크기: {df.shape}")
print("\n기본 통계:")
print(df.describe())
print()

# 3. 데이터 시각화
print("3. 데이터 시각화")
print("-" * 20)

plt.figure(figsize=(15, 5))

# 원본 데이터 산점도
plt.subplot(1, 3, 1)
plt.scatter(df['공부시간'], df['성적'], alpha=0.6, color='blue')
plt.xlabel('공부시간 (시간)')
plt.ylabel('성적 (점)')
plt.title('공부시간 vs 성적')
plt.grid(True, alpha=0.3)

# 4. 선형 회귀 모델 훈련
print("4. 선형 회귀 모델 훈련")
print("-" * 25)

# 데이터 분할 (훈련:테스트 = 8:2)
X = df[['공부시간']]  # 특성 (2차원 배열)
y = df['성적']        # 타겟 (1차원 배열)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"훈련 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")

# 선형 회귀 모델 생성 및 훈련
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 예측
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

print(f"\n📈 모델 계수:")
print(f"절편 (b): {linear_model.intercept_:.2f}")
print(f"기울기 (w): {linear_model.coef_[0]:.2f}")
print(f"📝 회귀식: 성적 = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f} × 공부시간")
print()

# 5. 모델 성능 평가
print("5. 모델 성능 평가")
print("-" * 20)

# 훈련 데이터 성능
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# 테스트 데이터 성능
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"📊 훈련 데이터 성능:")
print(f"  MSE (평균제곱오차): {train_mse:.2f}")
print(f"  R² (결정계수): {train_r2:.3f}")

print(f"\n📊 테스트 데이터 성능:")
print(f"  MSE: {test_mse:.2f}")
print(f"  R²: {test_r2:.3f}")
print(f"  RMSE: {np.sqrt(test_mse):.2f}")

print(f"\n💡 해석:")
print(f"  - R² = {test_r2:.3f} → 성적 변동의 {test_r2*100:.1f}%를 공부시간으로 설명 가능")
print(f"  - RMSE = {np.sqrt(test_mse):.2f} → 평균적으로 ±{np.sqrt(test_mse):.1f}점 오차")
print()

# 회귀선 시각화
plt.subplot(1, 3, 2)
plt.scatter(X_train, y_train, alpha=0.6, label='훈련 데이터', color='blue')
plt.scatter(X_test, y_test, alpha=0.6, label='테스트 데이터', color='red')

# 회귀선 그리기
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = linear_model.predict(x_line)
plt.plot(x_line, y_line, 'g-', linewidth=2, label='회귀선')

plt.xlabel('공부시간 (시간)')
plt.ylabel('성적 (점)')
plt.title(f'선형 회귀 (R² = {test_r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. 다항 회귀 (고급)
print("6. 다항 회귀 실습")
print("-" * 20)

# 2차 다항식 특성 생성
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 다항 회귀 모델 훈련
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# 예측
y_test_poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, y_test_poly_pred)

print(f"다항 회귀 성능:")
print(f"  R²: {poly_r2:.3f}")
print(f"  개선도: {poly_r2 - test_r2:.3f}")

# 다항 회귀선 시각화
plt.subplot(1, 3, 3)
plt.scatter(X_test, y_test, alpha=0.6, label='테스트 데이터', color='red')

x_line_poly = poly_features.transform(x_line)
y_line_poly = poly_model.predict(x_line_poly)
plt.plot(x_line, y_line_poly, 'purple', linewidth=2, label='다항 회귀선')
plt.plot(x_line, y_line, 'g--', alpha=0.7, label='선형 회귀선')

plt.xlabel('공부시간 (시간)')
plt.ylabel('성적 (점)')
plt.title(f'다항 회귀 (R² = {poly_r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 실제 예측 해보기
print("7. 실제 예측 실습")
print("-" * 20)

test_hours = [3, 5, 7, 9]
print("새로운 데이터로 성적 예측:")
for hours in test_hours:
    pred_score = linear_model.predict([[hours]])[0]
    print(f"  {hours}시간 공부 → 예상 성적: {pred_score:.1f}점")

print()

# 8. 잔차 분석 (모델 진단)
print("8. 잔차 분석 (모델 진단)")
print("-" * 25)

# 잔차 계산
residuals = y_test - y_test_pred

plt.figure(figsize=(12, 4))

# 잔차 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('예측값')
plt.ylabel('잔차 (실제값 - 예측값)')
plt.title('잔차 플롯')
plt.grid(True, alpha=0.3)

# 잔차 히스토그램
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.title('잔차 분포')
plt.axvline(0, color='r', linestyle='--')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"잔차 통계:")
print(f"  평균: {residuals.mean():.3f} (0에 가까울수록 좋음)")
print(f"  표준편차: {residuals.std():.3f}")

print("\n=== 회귀 분석 기초 학습 완료! ===")
print("💡 다음 주차에서는 분류 문제를 다뤄보겠습니다!")
