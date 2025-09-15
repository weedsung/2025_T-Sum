"""
Week 01: 파이썬 기초 다지기 - 강의 예제
"""

print("=== 파이썬 기초 다지기 ===\n")

# 1. 기본 문법과 변수
print("1. 기본 문법과 변수")
print("-" * 30)

# 변수와 자료형
name = "멘토링"
age = 25
height = 175.5
is_student = True

print(f"이름: {name} (타입: {type(name).__name__})")
print(f"나이: {age} (타입: {type(age).__name__})")
print(f"키: {height} (타입: {type(height).__name__})")
print(f"학생여부: {is_student} (타입: {type(is_student).__name__})")
print()

# 2. 조건문 (if, elif, else)
print("2. 조건문 예제")
print("-" * 30)

def check_grade(score):
    """점수에 따른 학점 판정 함수"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# 조건문 실습
test_scores = [95, 82, 77, 65, 45]
for score in test_scores:
    grade = check_grade(score)
    print(f"점수 {score}점 → 학점: {grade}")
print()

# 3. 반복문 (for, while)
print("3. 반복문 예제")
print("-" * 30)

# for문과 range()
print("1부터 10까지의 합:")
total = 0
for i in range(1, 11):
    total += i
print(f"합계: {total}")

# while문 예제
print("\n구구단 3단:")
i = 1
while i <= 9:
    print(f"3 × {i} = {3 * i}")
    i += 1
print()

# 4. 함수 정의와 활용
print("4. 함수 예제")
print("-" * 30)

def calculate_bmi(weight, height_m):
    """BMI 계산 함수"""
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)

def bmi_category(bmi):
    """BMI 카테고리 판정"""
    if bmi < 18.5:
        return "저체중"
    elif bmi < 25:
        return "정상"
    elif bmi < 30:
        return "과체중"
    else:
        return "비만"

# 함수 사용 예제
weight = 70
height_m = 1.75
bmi = calculate_bmi(weight, height_m)
category = bmi_category(bmi)

print(f"체중: {weight}kg, 키: {height_m}m")
print(f"BMI: {bmi}, 분류: {category}")
print()

# 5. 리스트 (List) 활용
print("5. 리스트 활용 예제")
print("-" * 30)

# 리스트 생성과 조작
fruits = ["사과", "바나나", "오렌지", "포도"]
print(f"과일 목록: {fruits}")

# 리스트 메소드들
fruits.append("딸기")
print(f"딸기 추가 후: {fruits}")

fruits.insert(1, "키위")
print(f"키위 삽입 후: {fruits}")

removed_fruit = fruits.pop()
print(f"{removed_fruit} 제거 후: {fruits}")

# 리스트 슬라이싱
print(f"처음 3개: {fruits[:3]}")
print(f"마지막 2개: {fruits[-2:]}")
print()

# 6. 딕셔너리 (Dictionary) 활용
print("6. 딕셔너리 활용 예제")
print("-" * 30)

# 학생 정보 딕셔너리
student = {
    "이름": "김멘티",
    "나이": 20,
    "전공": "컴퓨터공학",
    "성적": {"수학": 85, "영어": 92, "파이썬": 98}
}

print("학생 정보:")
for key, value in student.items():
    if key == "성적":
        print(f"{key}:")
        for subject, score in value.items():
            print(f"  {subject}: {score}점")
    else:
        print(f"{key}: {value}")

# 딕셔너리 메소드 활용
subjects = student["성적"]
average = sum(subjects.values()) / len(subjects)
print(f"\n평균 점수: {average:.1f}점")
print()

# 7. 종합 예제: 간단한 학생 관리 시스템
print("7. 종합 예제: 학생 관리 시스템")
print("-" * 30)

class StudentManager:
    """간단한 학생 관리 클래스"""
    
    def __init__(self):
        self.students = []
    
    def add_student(self, name, subjects_scores):
        """학생 추가"""
        student = {
            "이름": name,
            "과목점수": subjects_scores,
            "평균": sum(subjects_scores.values()) / len(subjects_scores)
        }
        self.students.append(student)
    
    def get_top_student(self):
        """최고 점수 학생 반환"""
        if not self.students:
            return None
        return max(self.students, key=lambda x: x["평균"])
    
    def display_all(self):
        """모든 학생 정보 출력"""
        for student in self.students:
            print(f"이름: {student['이름']}, 평균: {student['평균']:.1f}점")

# 학생 관리 시스템 사용
manager = StudentManager()

# 학생 데이터 추가
students_data = [
    ("김철수", {"수학": 85, "영어": 90, "과학": 78}),
    ("이영희", {"수학": 92, "영어": 88, "과학": 95}),
    ("박민수", {"수학": 76, "영어": 82, "과학": 80})
]

for name, scores in students_data:
    manager.add_student(name, scores)

print("전체 학생 목록:")
manager.display_all()

top_student = manager.get_top_student()
if top_student:
    print(f"\n최고 성적 학생: {top_student['이름']} ({top_student['평균']:.1f}점)")

print("\n=== 1주차 학습 완료! ===")
