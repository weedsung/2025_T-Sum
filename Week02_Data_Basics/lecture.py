"""
Week 02: 데이터 다루기 기초 - Pandas 강의 예제
"""

import pandas as pd
import numpy as np

print("=== 데이터 다루기 기초 - Pandas ===\n")

# 1. Pandas 소개와 기본 데이터 구조
print("1. Pandas 소개와 기본 데이터 구조")
print("-" * 40)

# Series 생성 (1차원 데이터)
print("📊 Series 예제:")
fruits = pd.Series(['사과', '바나나', '오렌지', '포도'])
prices = pd.Series([1000, 500, 800, 1200], index=['사과', '바나나', '오렌지', '포도'])

print("과일 목록:")
print(fruits)
print("\n과일 가격:")
print(prices)
print(f"사과 가격: {prices['사과']}원")
print()

# DataFrame 생성 (2차원 데이터)
print("📋 DataFrame 예제:")
student_data = {
    '이름': ['김철수', '이영희', '박민수'],
    '수학': [85, 92, 76],
    '영어': [90, 88, 82],
    '과학': [78, 95, 80]
}
df = pd.DataFrame(student_data)
print("학생 성적 DataFrame:")
print(df)
print()

# 2. CSV 파일 읽기와 기본 탐색
print("2. CSV 파일 읽기와 기본 탐색")
print("-" * 40)

# CSV 파일 읽기
try:
    grades_df = pd.read_csv('../data/student_grades.csv')
    print("✅ 학생 성적 데이터 불러오기 성공!")
    
    # 기본 정보 확인
    print(f"데이터 크기: {grades_df.shape} (행: {grades_df.shape[0]}, 열: {grades_df.shape[1]})")
    print(f"열 이름: {list(grades_df.columns)}")
    print()
    
    # 처음 5행 보기
    print("📝 처음 5행 데이터:")
    print(grades_df.head())
    print()
    
    # 마지막 3행 보기
    print("📝 마지막 3행 데이터:")
    print(grades_df.tail(3))
    print()
    
    # 데이터 정보 요약
    print("📊 데이터 정보 요약:")
    print(grades_df.info())
    print()
    
    # 기본 통계 정보
    print("📈 기본 통계 정보:")
    print(grades_df.describe())
    print()

except FileNotFoundError:
    print("❌ CSV 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    # 대신 예시 데이터 생성
    grades_df = pd.DataFrame({
        '이름': ['김철수', '이영희', '박민수', '최지윤', '정하늘'],
        '수학': [85, 92, 76, 96, 88],
        '영어': [90, 88, 82, 94, 86],
        '과학': [78, 95, 80, 92, 84]
    })
    print("🔄 예시 데이터로 진행합니다:")
    print(grades_df)
    print()

# 3. 데이터 선택과 필터링
print("3. 데이터 선택과 필터링")
print("-" * 40)

# 열 선택
print("🔍 특정 열 선택:")
print("수학 점수만 선택:")
print(grades_df['수학'])
print()

print("여러 열 선택 (수학, 영어):")
selected_columns = grades_df[['이름', '수학', '영어']]
print(selected_columns)
print()

# 행 선택 (인덱스 기반)
print("🎯 특정 행 선택:")
print("첫 번째 학생 정보:")
print(grades_df.iloc[0])
print()

print("처음 3명 학생 정보:")
print(grades_df.iloc[:3])
print()

# 조건부 필터링
print("🔎 조건부 필터링:")
print("수학 점수가 85점 이상인 학생:")
high_math_students = grades_df[grades_df['수학'] >= 85]
print(high_math_students)
print()

print("수학과 영어 모두 80점 이상인 학생:")
high_both = grades_df[(grades_df['수학'] >= 80) & (grades_df['영어'] >= 80)]
print(high_both)
print()

# 4. 데이터 조작과 분석
print("4. 데이터 조작과 분석")
print("-" * 40)

# 새로운 열 추가
print("➕ 새로운 열 추가:")
grades_df['평균'] = (grades_df['수학'] + grades_df['영어'] + grades_df['과학']) / 3
grades_df['평균'] = grades_df['평균'].round(2)
print("평균 점수 열 추가 후:")
print(grades_df[['이름', '평균']])
print()

# 정렬
print("📊 데이터 정렬:")
print("평균 점수 기준 내림차순 정렬:")
sorted_by_avg = grades_df.sort_values('평균', ascending=False)
print(sorted_by_avg[['이름', '평균']])
print()

# 그룹화와 집계
print("📈 통계 분석:")
print("과목별 평균 점수:")
subject_means = grades_df[['수학', '영어', '과학']].mean()
print(subject_means)
print()

print("과목별 최고/최저 점수:")
subject_stats = grades_df[['수학', '영어', '과학']].agg(['min', 'max', 'mean'])
print(subject_stats)
print()

# 5. 판매 데이터 분석 예제
print("5. 판매 데이터 분석 예제")
print("-" * 40)

# 판매 데이터 생성 (실제 파일이 없을 경우 대비)
sales_data = {
    '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03'],
    '제품명': ['노트북', '마우스', '모니터', '키보드', '태블릿'],
    '카테고리': ['전자제품', '전자제품', '전자제품', '전자제품', '전자제품'],
    '판매량': [5, 12, 3, 8, 4],
    '단가': [1200000, 35000, 450000, 85000, 800000],
    '매출액': [6000000, 420000, 1350000, 680000, 3200000]
}
sales_df = pd.DataFrame(sales_data)

print("🛒 판매 데이터:")
print(sales_df)
print()

# 카테고리별 분석
print("📊 카테고리별 매출 분석:")
category_sales = sales_df.groupby('카테고리')['매출액'].sum()
print(category_sales)
print()

# 일별 분석
print("📅 일별 총 매출:")
daily_sales = sales_df.groupby('날짜')['매출액'].sum()
print(daily_sales)
print()

# 베스트셀러 제품
print("🏆 매출액 기준 베스트셀러:")
best_seller = sales_df.loc[sales_df['매출액'].idxmax()]
print(f"제품명: {best_seller['제품명']}")
print(f"매출액: {best_seller['매출액']:,}원")
print()

# 6. 데이터 저장하기
print("6. 데이터 저장하기")
print("-" * 40)

# 분석 결과를 새로운 DataFrame으로 만들기
analysis_result = pd.DataFrame({
    '학생명': grades_df['이름'],
    '평균점수': grades_df['평균'],
    '등급': pd.cut(grades_df['평균'], 
                  bins=[0, 70, 80, 90, 100], 
                  labels=['D', 'C', 'B', 'A'])
})

print("📋 분석 결과:")
print(analysis_result)
print()

# CSV 파일로 저장 (예시)
try:
    analysis_result.to_csv('student_analysis_result.csv', index=False, encoding='utf-8')
    print("✅ 분석 결과가 'student_analysis_result.csv' 파일로 저장되었습니다.")
except Exception as e:
    print(f"❌ 파일 저장 중 오류: {e}")

print()

# 7. 유용한 Pandas 메소드들
print("7. 유용한 Pandas 메소드들")
print("-" * 40)

# 중복 제거
print("🔄 데이터 조작 메소드들:")
print(f"중복 행 개수: {grades_df.duplicated().sum()}")

# 결측치 확인
print(f"결측치 개수:\n{grades_df.isnull().sum()}")

# 데이터 타입 확인
print(f"\n데이터 타입:\n{grades_df.dtypes}")

# 유니크 값 개수
print(f"\n각 열의 유니크 값 개수:")
for col in grades_df.columns:
    if col != '이름':  # 숫자 열만
        unique_count = grades_df[col].nunique()
        print(f"{col}: {unique_count}개")

print()

# 8. 조건부 연산과 함수 적용
print("8. 조건부 연산과 함수 적용")
print("-" * 40)

# apply 함수 사용
def grade_to_letter(score):
    """점수를 등급으로 변환"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'D'

# 각 과목에 등급 적용
print("🏅 과목별 등급:")
for subject in ['수학', '영어', '과학']:
    grades_df[f'{subject}_등급'] = grades_df[subject].apply(grade_to_letter)

print(grades_df[['이름', '수학', '수학_등급', '영어', '영어_등급']])
print()

# 조건부 값 할당 (np.where 사용)
grades_df['우수학생'] = np.where(grades_df['평균'] >= 85, '우수', '일반')

print("🌟 우수학생 분류:")
print(grades_df[['이름', '평균', '우수학생']])

print("\n=== 2주차 Pandas 기초 학습 완료! ===")
print("💡 다음 주차에서는 데이터 전처리와 EDA(탐색적 데이터 분석)를 배워보겠습니다!")
