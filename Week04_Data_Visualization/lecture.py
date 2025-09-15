"""
Week 04: 데이터 시각화 - matplotlib & seaborn 기초
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== 데이터 시각화 기초 ===\n")

# 샘플 데이터 생성
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
temperature = 20 + 10 * np.sin(np.arange(30) * 2 * np.pi / 30) + np.random.normal(0, 2, 30)
humidity = 60 + np.random.normal(0, 10, 30)

# 1. 기본 선 그래프
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(dates, temperature, marker='o', linewidth=2, markersize=4)
plt.title('일별 기온 변화', fontsize=14, fontweight='bold')
plt.xlabel('날짜')
plt.ylabel('기온 (°C)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 2. 막대 그래프
cities = ['서울', '부산', '대구', '인천', '광주']
populations = [9.7, 3.4, 2.4, 2.9, 1.5]

plt.subplot(2, 2, 2)
bars = plt.bar(cities, populations, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
plt.title('도시별 인구 (백만명)', fontsize=14, fontweight='bold')
plt.ylabel('인구 (백만명)')

# 막대 위에 값 표시
for bar, pop in zip(bars, populations):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{pop}M', ha='center', va='bottom', fontweight='bold')

# 3. 산점도
plt.subplot(2, 2, 3)
plt.scatter(temperature, humidity, alpha=0.6, c=range(len(temperature)), cmap='viridis')
plt.xlabel('기온 (°C)')
plt.ylabel('습도 (%)')
plt.title('기온 vs 습도 관계', fontsize=14, fontweight='bold')
plt.colorbar(label='날짜 순서')

# 4. 히스토그램
plt.subplot(2, 2, 4)
plt.hist(temperature, bins=10, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('기온 (°C)')
plt.ylabel('빈도')
plt.title('기온 분포', fontsize=14, fontweight='bold')
plt.axvline(temperature.mean(), color='red', linestyle='--', 
           label=f'평균: {temperature.mean():.1f}°C')
plt.legend()

plt.tight_layout()
plt.show()

# Seaborn 고급 시각화
print("\n=== Seaborn 고급 시각화 ===")

# 학생 성적 데이터 생성
subjects = ['수학', '영어', '과학', '사회', '국어']
students = [f'학생{i+1}' for i in range(20)]

# 성적 데이터 생성
scores_data = []
for student in students:
    for subject in subjects:
        score = np.random.normal(75, 15)  # 평균 75, 표준편차 15
        score = max(0, min(100, score))  # 0-100 범위로 제한
        scores_data.append({
            '학생': student,
            '과목': subject,
            '점수': score,
            '학년': np.random.choice([1, 2, 3])
        })

df = pd.DataFrame(scores_data)

# 1. 박스플롯
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.boxplot(data=df, x='과목', y='점수')
plt.title('과목별 점수 분포 (박스플롯)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)

# 2. 바이올린플롯
plt.subplot(2, 3, 2)
sns.violinplot(data=df, x='학년', y='점수', hue='과목')
plt.title('학년별 과목별 점수 분포', fontsize=12, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. 히트맵
plt.subplot(2, 3, 3)
pivot_df = df.pivot_table(values='점수', index='학년', columns='과목', aggfunc='mean')
sns.heatmap(pivot_df, annot=True, cmap='RdYlBu_r', fmt='.1f')
plt.title('학년별 과목별 평균 점수', fontsize=12, fontweight='bold')

# 4. 분포도 (distplot 대신 histplot 사용)
plt.subplot(2, 3, 4)
for subject in subjects[:3]:  # 처음 3과목만
    subject_scores = df[df['과목'] == subject]['점수']
    sns.histplot(subject_scores, alpha=0.6, label=subject, kde=True)
plt.xlabel('점수')
plt.ylabel('빈도')
plt.title('과목별 점수 분포 비교', fontsize=12, fontweight='bold')
plt.legend()

# 5. 쌍별 관계도 (pairplot을 위한 데이터 변환)
plt.subplot(2, 3, 5)
pivot_scores = df.pivot_table(values='점수', index='학생', columns='과목', aggfunc='mean')
correlation = pivot_scores.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('과목간 상관관계', fontsize=12, fontweight='bold')

# 6. 시계열 그래프 (기온 데이터)
plt.subplot(2, 3, 6)
sns.lineplot(x=range(len(temperature)), y=temperature, marker='o')
plt.fill_between(range(len(temperature)), temperature, alpha=0.3)
plt.xlabel('날짜')
plt.ylabel('기온 (°C)')
plt.title('기온 변화 추이', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ 데이터 시각화 기본 기법들을 학습했습니다!")
print("💡 다음 실습에서는 공공데이터를 활용한 실제 시각화를 진행해보겠습니다.")
