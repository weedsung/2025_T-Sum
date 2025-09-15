"""
Week 01 미니 프로젝트: 간단한 계산기 프로그램
함수와 반복문을 활용한 종합 실습
"""

def add(x, y):
    """덧셈 함수"""
    return x + y

def subtract(x, y):
    """뺄셈 함수"""
    return x - y

def multiply(x, y):
    """곱셈 함수"""
    return x * y

def divide(x, y):
    """나눗셈 함수"""
    if y == 0:
        return "오류: 0으로 나눌 수 없습니다."
    return x / y

def power(x, y):
    """거듭제곱 함수"""
    return x ** y

def square_root(x):
    """제곱근 함수"""
    if x < 0:
        return "오류: 음수의 제곱근은 계산할 수 없습니다."
    return x ** 0.5

def show_menu():
    """메뉴 출력 함수"""
    print("\n=== 간단한 계산기 ===")
    print("1. 덧셈 (+)")
    print("2. 뺄셈 (-)")
    print("3. 곱셈 (×)")
    print("4. 나눗셈 (÷)")
    print("5. 거듭제곱 (^)")
    print("6. 제곱근 (√)")
    print("7. 계산 기록 보기")
    print("8. 프로그램 종료")
    print("-" * 25)

def get_number(prompt):
    """숫자 입력 함수 (에러 처리 포함)"""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("올바른 숫자를 입력해주세요.")

def format_result(result):
    """결과 포맷팅 함수"""
    if isinstance(result, str):  # 오류 메시지인 경우
        return result
    elif result == int(result):  # 정수인 경우
        return str(int(result))
    else:  # 실수인 경우
        return f"{result:.6f}".rstrip('0').rstrip('.')

def main():
    """메인 계산기 프로그램"""
    print("🔢 파이썬 계산기에 오신 것을 환영합니다!")
    
    # 계산 기록을 저장할 리스트
    history = []
    
    while True:
        show_menu()
        
        try:
            choice = int(input("원하는 기능을 선택하세요 (1-8): "))
        except ValueError:
            print("올바른 번호를 입력해주세요.")
            continue
        
        if choice == 8:
            print("계산기를 종료합니다. 안녕히가세요! 👋")
            break
        
        elif choice == 7:
            # 계산 기록 보기
            if history:
                print("\n=== 계산 기록 ===")
                for i, record in enumerate(history, 1):
                    print(f"{i}. {record}")
            else:
                print("계산 기록이 없습니다.")
            continue
        
        elif choice in [1, 2, 3, 4, 5]:
            # 두 개의 숫자가 필요한 연산
            num1 = get_number("첫 번째 숫자를 입력하세요: ")
            num2 = get_number("두 번째 숫자를 입력하세요: ")
            
            if choice == 1:
                result = add(num1, num2)
                operation = f"{format_result(num1)} + {format_result(num2)} = {format_result(result)}"
            elif choice == 2:
                result = subtract(num1, num2)
                operation = f"{format_result(num1)} - {format_result(num2)} = {format_result(result)}"
            elif choice == 3:
                result = multiply(num1, num2)
                operation = f"{format_result(num1)} × {format_result(num2)} = {format_result(result)}"
            elif choice == 4:
                result = divide(num1, num2)
                if isinstance(result, str):
                    operation = f"{format_result(num1)} ÷ {format_result(num2)} = {result}"
                else:
                    operation = f"{format_result(num1)} ÷ {format_result(num2)} = {format_result(result)}"
            elif choice == 5:
                result = power(num1, num2)
                operation = f"{format_result(num1)} ^ {format_result(num2)} = {format_result(result)}"
        
        elif choice == 6:
            # 제곱근 (하나의 숫자만 필요)
            num = get_number("제곱근을 구할 숫자를 입력하세요: ")
            result = square_root(num)
            if isinstance(result, str):
                operation = f"√{format_result(num)} = {result}"
            else:
                operation = f"√{format_result(num)} = {format_result(result)}"
        
        else:
            print("올바른 번호를 선택해주세요.")
            continue
        
        # 결과 출력 및 기록 저장
        print(f"\n결과: {operation}")
        history.append(operation)
        
        # 계속 계산할지 묻기
        continue_calc = input("\n다른 계산을 하시겠습니까? (y/n): ").lower().strip()
        if continue_calc in ['n', 'no', '아니오', 'ㄴ']:
            print("계산기를 종료합니다. 안녕히가세요! 👋")
            break

# 추가 기능: 고급 계산기 클래스
class AdvancedCalculator:
    """고급 계산기 클래스 (객체지향 방식)"""
    
    def __init__(self):
        self.history = []
        self.memory = 0  # 메모리 기능
    
    def calculate(self, operation, x, y=None):
        """통합 계산 함수"""
        operations = {
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'multiply': lambda a, b: a * b,
            'divide': lambda a, b: a / b if b != 0 else "0으로 나눌 수 없습니다",
            'power': lambda a, b: a ** b,
            'sqrt': lambda a, b=None: a ** 0.5 if a >= 0 else "음수의 제곱근 불가"
        }
        
        if operation in operations:
            if operation == 'sqrt':
                result = operations[operation](x)
            else:
                result = operations[operation](x, y)
            
            # 기록 저장
            if operation == 'sqrt':
                record = f"√{x} = {result}"
            else:
                symbols = {'add': '+', 'subtract': '-', 'multiply': '×', 
                          'divide': '÷', 'power': '^'}
                record = f"{x} {symbols[operation]} {y} = {result}"
            
            self.history.append(record)
            return result
        else:
            return "알 수 없는 연산입니다"
    
    def save_to_memory(self, value):
        """메모리에 값 저장"""
        self.memory = value
        print(f"메모리에 {value}를 저장했습니다.")
    
    def recall_memory(self):
        """메모리 값 불러오기"""
        return self.memory
    
    def clear_memory(self):
        """메모리 초기화"""
        self.memory = 0
        print("메모리를 초기화했습니다.")

def demo_advanced_calculator():
    """고급 계산기 데모"""
    calc = AdvancedCalculator()
    
    print("\n=== 고급 계산기 데모 ===")
    
    # 몇 가지 계산 실행
    result1 = calc.calculate('add', 10, 5)
    print(f"10 + 5 = {result1}")
    
    result2 = calc.calculate('multiply', result1, 2)
    print(f"{result1} × 2 = {result2}")
    
    calc.save_to_memory(result2)
    
    result3 = calc.calculate('sqrt', 16)
    print(f"√16 = {result3}")
    
    print(f"\n메모리에 저장된 값: {calc.recall_memory()}")
    print("계산 기록:")
    for i, record in enumerate(calc.history, 1):
        print(f"{i}. {record}")

if __name__ == "__main__":
    print("어떤 계산기를 사용하시겠습니까?")
    print("1. 기본 계산기")
    print("2. 고급 계산기 데모")
    
    choice = input("선택하세요 (1 또는 2): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        demo_advanced_calculator()
    else:
        print("기본 계산기를 실행합니다.")
        main()
