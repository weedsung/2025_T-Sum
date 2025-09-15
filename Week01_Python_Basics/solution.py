"""
Week 01: 파이썬 기초 실습 문제 해답
"""

print("=== 파이썬 기초 실습 문제 해답 ===\n")

# 문제 1 해답: 변수와 연산
print("문제 1 해답: 변수와 연산")
print("-" * 30)

def basic_calculator():
    """기본 사칙연산 계산기"""
    try:
        num1 = float(input("첫 번째 숫자를 입력하세요: "))
        num2 = float(input("두 번째 숫자를 입력하세요: "))
        
        print(f"덧셈: {num1 + num2}")
        print(f"뺄셈: {num1 - num2}")
        print(f"곱셈: {num1 * num2}")
        if num2 != 0:
            print(f"나눗셈: {num1 / num2:.2f}")
        else:
            print("나눗셈: 0으로 나눌 수 없습니다.")
    except ValueError:
        print("올바른 숫자를 입력해주세요.")

# 예시 실행 (입력 없이 보여주기)
num1, num2 = 10, 3
print(f"예시: {num1}, {num2}")
print(f"덧셈: {num1 + num2}")
print(f"뺄셈: {num1 - num2}")
print(f"곱셈: {num1 * num2}")
print(f"나눗셈: {num1 / num2:.2f}")
print()

# 문제 2 해답: 조건문 활용
print("문제 2 해답: 조건문 활용")
print("-" * 30)

def classify_age(age):
    """나이에 따른 연령대 분류"""
    if 0 <= age <= 12:
        return "어린이"
    elif 13 <= age <= 19:
        return "청소년"
    elif 20 <= age <= 64:
        return "성인"
    elif age >= 65:
        return "노인"
    else:
        return "올바른 나이를 입력해주세요"

# 예시 실행
test_ages = [8, 15, 25, 70]
for age in test_ages:
    category = classify_age(age)
    print(f"{age}세 → {category}")
print()

# 문제 3 해답: 반복문 활용
print("문제 3 해답: 반복문 활용")
print("-" * 30)

# 1) 3의 배수의 합
sum_of_multiples_3 = 0
for i in range(1, 101):
    if i % 3 == 0:
        sum_of_multiples_3 += i
print(f"1) 3의 배수의 합: {sum_of_multiples_3}")

# 2) 5의 배수의 개수
count_multiples_5 = 0
for i in range(1, 101):
    if i % 5 == 0:
        count_multiples_5 += 1
print(f"2) 5의 배수의 개수: {count_multiples_5}개")

# 3) 3의 배수이면서 5의 배수인 수 (15의 배수)
multiples_15 = []
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        multiples_15.append(i)
print(f"3) 3과 5의 공배수: {multiples_15}")
print()

# 문제 4 해답: 함수 정의
print("문제 4 해답: 함수 정의")
print("-" * 30)

def is_even(number):
    """짝수 홀수 판별 함수"""
    return number % 2 == 0

def find_max(numbers):
    """리스트 최댓값 찾기 함수"""
    if not numbers:
        return None
    
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value

def reverse_string(text):
    """문자열 뒤집기 함수"""
    return text[::-1]
    # 또는 다음과 같이 반복문 사용:
    # reversed_text = ""
    # for char in text:
    #     reversed_text = char + reversed_text
    # return reversed_text

# 함수 테스트
print(f"5는 짝수인가? {is_even(5)}")
print(f"8은 짝수인가? {is_even(8)}")
print(f"[3, 7, 2, 9, 1]의 최댓값: {find_max([3, 7, 2, 9, 1])}")
print(f"'파이썬'을 뒤집으면: '{reverse_string('파이썬')}'")
print()

# 문제 5 해답: 리스트 조작
print("문제 5 해답: 리스트 조작")
print("-" * 30)

def list_manipulation_demo():
    """리스트 조작 예시"""
    # 예시 데이터로 시연
    numbers = [85, 92, 78, 96, 88]
    
    print(f"입력된 숫자들: {numbers}")
    
    # 합과 평균 계산
    total = sum(numbers)
    average = total / len(numbers)
    print(f"합: {total}, 평균: {average:.1f}")
    
    # 정렬
    sorted_numbers = sorted(numbers)
    print(f"정렬된 리스트: {sorted_numbers}")
    
    # 최댓값, 최솟값
    print(f"가장 큰 값: {max(numbers)}")
    print(f"가장 작은 값: {min(numbers)}")

list_manipulation_demo()
print()

# 문제 6 해답: 딕셔너리 활용
print("문제 6 해답: 딕셔너리 활용")
print("-" * 30)

def student_grade_manager():
    """학생 성적 관리 프로그램"""
    # 학생 성적 딕셔너리
    students = {
        "김철수": 85,
        "이영희": 92,
        "박민수": 78,
        "최지윤": 96,
        "정하늘": 88
    }
    
    print("학생 성적 정보:")
    for name, score in students.items():
        print(f"{name}: {score}점")
    
    # 평균 점수 계산
    average = sum(students.values()) / len(students)
    print(f"\n평균 점수: {average:.1f}점")
    
    # 가장 높은 점수와 학생
    top_student = max(students, key=students.get)
    top_score = students[top_student]
    print(f"최고 점수: {top_student} ({top_score}점)")
    
    # 80점 이상 학생들
    excellent_students = [name for name, score in students.items() if score >= 80]
    print(f"80점 이상 학생들: {', '.join(excellent_students)}")

student_grade_manager()
print()

# 문제 7 해답: 종합 문제 (가계부)
print("문제 7 해답: 종합 문제 (가계부)")
print("-" * 30)

class SimpleAccountBook:
    """간단한 가계부 클래스"""
    
    def __init__(self):
        self.transactions = []
    
    def add_transaction(self, transaction_type, amount, description):
        """거래 내역 추가"""
        transaction = {
            '종류': transaction_type,
            '금액': amount,
            '내용': description
        }
        self.transactions.append(transaction)
        print(f"{transaction_type} {amount}원 ({description}) 추가되었습니다.")
    
    def show_all_transactions(self):
        """전체 내역 출력"""
        if not self.transactions:
            print("거래 내역이 없습니다.")
            return
        
        print("\n=== 전체 거래 내역 ===")
        for i, transaction in enumerate(self.transactions, 1):
            print(f"{i}. {transaction['종류']} {transaction['금액']:,}원 - {transaction['내용']}")
    
    def calculate_balance(self):
        """현재 잔액 계산"""
        balance = 0
        for transaction in self.transactions:
            if transaction['종류'] == '수입':
                balance += transaction['금액']
            else:  # 지출
                balance -= transaction['금액']
        return balance
    
    def run(self):
        """가계부 프로그램 실행"""
        print("=== 간단한 가계부 프로그램 ===")
        
        while True:
            print("\n1. 수입 추가")
            print("2. 지출 추가")
            print("3. 전체 내역 보기")
            print("4. 현재 잔액 확인")
            print("5. 프로그램 종료")
            
            choice = input("선택하세요 (1-5): ").strip()
            
            if choice == '1':
                amount = int(input("수입 금액: "))
                description = input("수입 내용: ")
                self.add_transaction('수입', amount, description)
            
            elif choice == '2':
                amount = int(input("지출 금액: "))
                description = input("지출 내용: ")
                self.add_transaction('지출', amount, description)
            
            elif choice == '3':
                self.show_all_transactions()
            
            elif choice == '4':
                balance = self.calculate_balance()
                print(f"현재 잔액: {balance:,}원")
            
            elif choice == '5':
                print("프로그램을 종료합니다.")
                break
            
            else:
                print("올바른 번호를 선택해주세요.")

# 가계부 예시 실행 (데모 데이터)
def demo_account_book():
    """가계부 데모"""
    account_book = SimpleAccountBook()
    
    # 예시 데이터 추가
    account_book.add_transaction('수입', 1000000, '월급')
    account_book.add_transaction('지출', 500000, '월세')
    account_book.add_transaction('지출', 200000, '식비')
    account_book.add_transaction('수입', 50000, '용돈')
    
    account_book.show_all_transactions()
    
    balance = account_book.calculate_balance()
    print(f"\n현재 잔액: {balance:,}원")

print("가계부 프로그램 데모:")
demo_account_book()

print("\n=== 모든 문제 해답 완료! ===")
print("실제 프로그램을 실행하려면 각 함수를 호출하거나")
print("account_book = SimpleAccountBook(); account_book.run() 을 실행하세요.")
