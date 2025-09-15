"""
Week 02: 데이터 다루기 기초 실습 문제 해답
"""

import pandas as pd
import numpy as np

print("=== 데이터 다루기 기초 실습 문제 해답 ===\n")

# 실습용 데이터 생성
student_data = {
    '이름': ['김철수', '이영희', '박민수', '최지윤', '정하늘', '오세진', '윤미래', '장동건'],
    '학년': [1, 2, 1, 3, 2, 1, 3, 2],
    '수학': [85, 92, 76, 96, 88, 79, 94, 82],
    '영어': [90, 88, 82, 94, 86, 91, 89, 78],
    '과학': [78, 95, 80, 92, 84, 77, 91, 85],
    '출석일수': [180, 175, 182, 178, 179, 181, 177, 183]
}
df = pd.DataFrame(student_data)

# 문제 1 해답: 기본 데이터 탐색
print("문제 1 해답: 기본 데이터 탐색")
print("-" * 30)

print(f"1) 데이터 크기: {df.shape} (행: {df.shape[0]}개, 열: {df.shape[1]}개)")
print(f"2) 열 이름: {list(df.columns)}")
print("3) 데이터 타입:")
print(df.dtypes)
print("\n4) 수치형 열들의 기본 통계:")
print(df.describe())
print()

# 문제 2 해답: 데이터 선택과 필터링
print("문제 2 해답: 데이터 선택과 필터링")
print("-" * 30)

print("1) 수학 점수만:")
print(df['수학'])
print("\n2) 이름과 수학 점수:")
print(df[['이름', '수학']])
print("\n3) 수학 85점 이상 학생:")
print(df[df['수학'] >= 85])
print("\n4) 2학년 학생들:")
print(df[df['학년'] == 2])
print("\n5) 수학과 영어 모두 80점 이상:")
print(df[(df['수학'] >= 80) & (df['영어'] >= 80)])
print()

# 문제 3 해답: 새로운 열 추가와 계산
print("문제 3 해답: 새로운 열 추가와 계산")
print("-" * 30)

# 평균과 총점 계산
df['평균'] = ((df['수학'] + df['영어'] + df['과학']) / 3).round(2)
df['총점'] = df['수학'] + df['영어'] + df['과학']

# 등급 계산
def calculate_grade(avg):
    if avg >= 90:
        return 'A'
    elif avg >= 80:
        return 'B'
    elif avg >= 70:
        return 'C'
    else:
        return 'D'

df['등급'] = df['평균'].apply(calculate_grade)

# 출석률 계산
df['출석률'] = ((df['출석일수'] / 185) * 100).round(1)

print("새로운 열 추가 결과:")
print(df[['이름', '평균', '총점', '등급', '출석률']])
print()

# 문제 4 해답: 데이터 정렬과 순위
print("문제 4 해답: 데이터 정렬과 순위")
print("-" * 30)

print("1) 평균 점수 내림차순 정렬:")
sorted_by_avg = df.sort_values('평균', ascending=False)
print(sorted_by_avg[['이름', '학년', '평균']])

print("\n2) 학년별, 평균점수별 정렬:")
sorted_by_grade_avg = df.sort_values(['학년', '평균'], ascending=[True, False])
print(sorted_by_grade_avg[['이름', '학년', '평균']])

print("\n3) 평균 점수 순위:")
df['순위'] = df['평균'].rank(ascending=False, method='min').astype(int)
print(df[['이름', '평균', '순위']].sort_values('순위'))
print()

# 문제 5 해답: 그룹별 분석
print("문제 5 해답: 그룹별 분석")
print("-" * 30)

print("1) 학년별 과목 평균:")
subject_avg_by_grade = df.groupby('학년')[['수학', '영어', '과학']].mean()
print(subject_avg_by_grade)

print("\n2) 학년별 학생 수:")
student_count = df.groupby('학년').size()
print(student_count)

print("\n3) 학년별 평균 출석률:")
attendance_by_grade = df.groupby('학년')['출석률'].mean()
print(attendance_by_grade)

print("\n4) 학년별 최고 점수 학생:")
top_students = df.loc[df.groupby('학년')['평균'].idxmax()]
print(top_students[['학년', '이름', '평균']])
print()

# 문제 6 해답: 조건부 연산
print("문제 6 해답: 조건부 연산")
print("-" * 30)

# 성취도 분류
df['성취도'] = np.where(df['평균'] >= 85, '우수',
                      np.where(df['평균'] >= 75, '양호', '보통'))

# 출석 분류
df['출석분류'] = np.where(df['출석률'] >= 95, '개근',
                        np.where(df['출석률'] >= 90, '우수', '일반'))

# 최고 점수 과목 찾기
def find_best_subject(row):
    subjects = {'수학': row['수학'], '영어': row['영어'], '과학': row['과학']}
    return max(subjects, key=subjects.get)

df['최고과목'] = df.apply(find_best_subject, axis=1)

print("조건부 연산 결과:")
print(df[['이름', '평균', '성취도', '출석률', '출석분류', '최고과목']])
print()

# 문제 7 해답: 데이터 요약과 통계
print("문제 7 해답: 데이터 요약과 통계")
print("-" * 30)

print("1) 과목별 통계:")
subject_stats = df[['수학', '영어', '과학']].agg(['mean', 'max', 'min'])
print(subject_stats)

print("\n2) 등급별 학생 수:")
grade_counts = df['등급'].value_counts().sort_index()
print(grade_counts)

print("\n3) 학년별 평균과 표준편차:")
grade_stats = df.groupby('학년')['평균'].agg(['mean', 'std'])
print(grade_stats)

print("\n4) 상위 30% 학생들의 평균:")
top_30_percent = df.nlargest(int(len(df) * 0.3), '평균')
print(f"상위 30% 학생들의 평균: {top_30_percent['평균'].mean():.2f}")
print()

# 문제 8 해답: 판매 데이터 분석
print("문제 8 해답: 판매 데이터 분석")
print("-" * 30)

sales_data = {
    '날짜': pd.date_range('2024-01-01', periods=10, freq='D'),
    '제품': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    '카테고리': ['전자', '가구', '전자', '의류', '가구', '전자', '의류', '가구', '전자', '의류'],
    '판매량': [5, 3, 7, 2, 4, 6, 3, 5, 8, 4],
    '단가': [10000, 50000, 10000, 30000, 50000, 10000, 30000, 50000, 10000, 30000]
}
sales_df = pd.DataFrame(sales_data)
sales_df['매출'] = sales_df['판매량'] * sales_df['단가']

print("1) 제품별 총 매출액:")
product_sales = sales_df.groupby('제품')['매출'].sum()
print(product_sales)

print("\n2) 카테고리별 평균 판매량:")
category_avg_sales = sales_df.groupby('카테고리')['판매량'].mean()
print(category_avg_sales)

print("\n3) 일별 총 매출액:")
daily_sales = sales_df.groupby('날짜')['매출'].sum()
print(daily_sales.head())

print("\n4) 가장 많이 팔린 제품:")
best_selling = sales_df.groupby('제품')['판매량'].sum().idxmax()
best_selling_amount = sales_df.groupby('제품')['판매량'].sum().max()
print(f"제품 {best_selling}: {best_selling_amount}개")

print("\n5) 매출이 가장 높았던 날짜:")
best_day = daily_sales.idxmax()
best_day_sales = daily_sales.max()
print(f"{best_day.strftime('%Y-%m-%d')}: {best_day_sales:,}원")

print("\n6) 전자 카테고리 평균 단가:")
electronics_avg_price = sales_df[sales_df['카테고리'] == '전자']['단가'].mean()
print(f"{electronics_avg_price:,}원")
print()

# 문제 9 해답: 데이터 저장하기
print("문제 9 해답: 데이터 저장하기")
print("-" * 30)

try:
    # 우수 학생 저장
    excellent_students = df[df['등급'] == 'A']
    excellent_students.to_csv('excellent_students.csv', index=False, encoding='utf-8')
    print("✅ excellent_students.csv 저장 완료")
    
    # 학년별 통계 저장
    grade_summary = df.groupby('학년').agg({
        '수학': 'mean',
        '영어': 'mean', 
        '과학': 'mean',
        '평균': 'mean',
        '출석률': 'mean'
    }).round(2)
    grade_summary.to_csv('grade_summary.csv', encoding='utf-8')
    print("✅ grade_summary.csv 저장 완료")
    
    # 전체 데이터 저장
    df.to_csv('complete_student_data.csv', index=False, encoding='utf-8')
    print("✅ complete_student_data.csv 저장 완료")
    
except Exception as e:
    print(f"❌ 파일 저장 중 오류: {e}")

print()

# 문제 10 해답: 고급 데이터 조작
print("문제 10 해답: 고급 데이터 조작")
print("-" * 30)

print("1) 학년별-과목별 평균 점수 피벗 테이블:")
pivot_table = pd.pivot_table(df, values=['수학', '영어', '과학'], 
                           index='학년', aggfunc='mean')
print(pivot_table)

print("\n2) 각 학생의 강점 과목:")
print(df[['이름', '최고과목']])

print("\n3) 과목간 상관관계:")
correlation = df[['수학', '영어', '과학']].corr()
print(correlation)

print("\n4) 평균 점수 4분위수 그룹:")
df['분위수그룹'] = pd.qcut(df['평균'], q=4, labels=['하위', '중하위', '중상위', '상위'])
quartile_summary = df.groupby('분위수그룹').size()
print(quartile_summary)

print("\n최종 데이터프레임:")
print(df[['이름', '학년', '평균', '등급', '성취도', '분위수그룹']])

print("\n=== 모든 문제 해답 완료! ===")
print("🎉 Pandas 기본 기능들을 모두 실습해보았습니다!")
print("💡 다음 주차에서는 데이터 전처리와 EDA를 더 깊이 있게 배워보겠습니다.")
