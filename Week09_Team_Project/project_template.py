"""
Week 09-10: 팀 프로젝트 템플릿
데이터 분석 프로젝트를 위한 표준 구조
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DataAnalysisProject:
    """데이터 분석 프로젝트 템플릿 클래스"""
    
    def __init__(self, project_name):
        self.project_name = project_name
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.results = {}
        
        print(f"🚀 프로젝트 시작: {project_name}")
        print("="*50)
    
    def load_data(self, file_path=None, data=None):
        """
        1단계: 데이터 로드
        """
        print("📂 1단계: 데이터 로드")
        print("-" * 20)
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                print(f"✅ 파일에서 데이터 로드 성공: {file_path}")
            except Exception as e:
                print(f"❌ 파일 로드 실패: {e}")
                return False
        elif data is not None:
            self.data = data
            print("✅ 직접 제공된 데이터 사용")
        else:
            # 예시 데이터 생성
            self.data = self._create_sample_data()
            print("✅ 샘플 데이터 생성")
        
        print(f"데이터 크기: {self.data.shape}")
        print(f"열 정보: {list(self.data.columns)}")
        print()
        return True
    
    def explore_data(self):
        """
        2단계: 탐색적 데이터 분석 (EDA)
        """
        print("🔍 2단계: 탐색적 데이터 분석 (EDA)")
        print("-" * 30)
        
        # 기본 정보
        print("📊 데이터 기본 정보:")
        print(self.data.info())
        print()
        
        # 기술 통계
        print("📈 기술 통계:")
        print(self.data.describe())
        print()
        
        # 결측치 확인
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("🔍 결측치 정보:")
            print(missing_values[missing_values > 0])
        else:
            print("✅ 결측치 없음")
        print()
        
        # 데이터 타입별 분석
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        print(f"📊 수치형 변수: {len(numeric_cols)}개")
        print(f"📝 범주형 변수: {len(categorical_cols)}개")
        print()
        
        return numeric_cols, categorical_cols
    
    def visualize_data(self):
        """
        3단계: 데이터 시각화
        """
        print("📊 3단계: 데이터 시각화")
        print("-" * 20)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # 상관관계 히트맵
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            correlation = self.data[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('변수간 상관관계')
            
            # 분포 히스토그램
            plt.subplot(2, 2, 2)
            self.data[numeric_cols[0]].hist(bins=20, alpha=0.7)
            plt.title(f'{numeric_cols[0]} 분포')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('빈도')
            
            # 산점도 (첫 두 변수)
            if len(numeric_cols) >= 2:
                plt.subplot(2, 2, 3)
                plt.scatter(self.data[numeric_cols[0]], self.data[numeric_cols[1]], alpha=0.6)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
            
            # 박스플롯
            plt.subplot(2, 2, 4)
            self.data[numeric_cols[:4]].boxplot()
            plt.title('박스플롯')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        print("✅ 시각화 완료")
        print()
    
    def preprocess_data(self, target_column):
        """
        4단계: 데이터 전처리
        """
        print("🔧 4단계: 데이터 전처리")
        print("-" * 20)
        
        # 타겟 변수 분리
        if target_column not in self.data.columns:
            print(f"❌ 타겟 변수 '{target_column}'를 찾을 수 없습니다.")
            return False
        
        self.y = self.data[target_column]
        self.X = self.data.drop(target_column, axis=1)
        
        print(f"✅ 타겟 변수: {target_column}")
        print(f"특성 변수: {len(self.X.columns)}개")
        
        # 범주형 변수 인코딩
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"🔤 범주형 변수 인코딩: {len(categorical_cols)}개")
            for col in categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        # 결측치 처리 (평균값으로 대체)
        if self.X.isnull().sum().sum() > 0:
            print("🔍 결측치 처리 중...")
            self.X = self.X.fillna(self.X.mean())
        
        # 타겟 변수가 범주형인 경우 인코딩
        if self.y.dtype == 'object':
            le_y = LabelEncoder()
            self.y = le_y.fit_transform(self.y)
            print("🎯 타겟 변수 인코딩 완료")
        
        print("✅ 전처리 완료")
        print()
        return True
    
    def train_model(self, model_type='classification'):
        """
        5단계: 모델 학습
        """
        print("🤖 5단계: 모델 학습")
        print("-" * 15)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"훈련 데이터: {X_train.shape[0]}개")
        print(f"테스트 데이터: {X_test.shape[0]}개")
        
        # 특성 정규화
        if model_type == 'classification':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 랜덤 포레스트 분류기
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # 예측
            y_pred = self.model.predict(X_test_scaled)
            
            # 평가
            accuracy = (y_pred == y_test).mean()
            
            self.results = {
                'model_type': 'classification',
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_importance': self.model.feature_importances_
            }
            
            print(f"✅ 분류 모델 학습 완료")
            print(f"정확도: {accuracy:.3f}")
        
        print()
        return True
    
    def evaluate_model(self):
        """
        6단계: 모델 평가
        """
        print("📊 6단계: 모델 평가")
        print("-" * 15)
        
        if not self.results:
            print("❌ 학습된 모델이 없습니다.")
            return
        
        if self.results['model_type'] == 'classification':
            # 분류 리포트
            print("📋 분류 리포트:")
            print(classification_report(self.results['y_test'], self.results['y_pred']))
            
            # 혼동 행렬 시각화
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            cm = confusion_matrix(self.results['y_test'], self.results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('혼동 행렬')
            plt.xlabel('예측')
            plt.ylabel('실제')
            
            # 특성 중요도
            plt.subplot(1, 2, 2)
            feature_importance = pd.DataFrame({
                '특성': self.X.columns,
                '중요도': self.results['feature_importance']
            }).sort_values('중요도', ascending=True)
            
            plt.barh(feature_importance['특성'], feature_importance['중요도'])
            plt.title('특성 중요도')
            plt.xlabel('중요도')
            
            plt.tight_layout()
            plt.show()
        
        print("✅ 모델 평가 완료")
        print()
    
    def generate_report(self):
        """
        7단계: 결과 보고서 생성
        """
        print("📝 7단계: 프로젝트 보고서")
        print("-" * 20)
        
        report = f"""
        
📊 {self.project_name} 분석 보고서
{"="*50}

📈 데이터 개요:
- 데이터 크기: {self.data.shape[0]}행 × {self.data.shape[1]}열
- 특성 변수: {len(self.X.columns)}개
- 타겟 변수: {self.y.name if hasattr(self.y, 'name') else '타겟'}

🤖 모델 성능:
- 모델 타입: {self.results.get('model_type', 'N/A')}
- 정확도: {self.results.get('accuracy', 0):.3f}

🔍 주요 인사이트:
- 가장 중요한 특성: {self.X.columns[np.argmax(self.results.get('feature_importance', [0]))] if 'feature_importance' in self.results else 'N/A'}
- 모델 복잡도: 적절함

📋 결론 및 제언:
1. 모델이 데이터의 패턴을 잘 학습함
2. 특성 엔지니어링을 통한 성능 개선 가능
3. 더 많은 데이터 수집 권장
4. 하이퍼파라미터 튜닝 고려

🚀 다음 단계:
- 모델 성능 개선
- 실제 서비스 배포 준비
- 지속적인 모니터링
        """
        
        print(report)
        return report
    
    def _create_sample_data(self):
        """샘플 데이터 생성"""
        np.random.seed(42)
        n_samples = 1000
        
        # 특성 생성
        age = np.random.randint(18, 65, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        experience = np.random.randint(0, 20, n_samples)
        education = np.random.choice(['고졸', '대졸', '대학원졸'], n_samples)
        
        # 타겟 생성 (구매 여부)
        purchase_prob = (age * 0.01 + income * 0.00001 + experience * 0.05 + 
                        (education == '대학원졸') * 0.2)
        purchase = np.random.binomial(1, np.clip(purchase_prob, 0, 1), n_samples)
        
        return pd.DataFrame({
            '나이': age,
            '소득': income,
            '경력': experience,
            '학력': education,
            '구매여부': purchase
        })

def run_sample_project():
    """샘플 프로젝트 실행"""
    
    # 프로젝트 초기화
    project = DataAnalysisProject("고객 구매 예측 프로젝트")
    
    # 1. 데이터 로드
    project.load_data()
    
    # 2. 탐색적 데이터 분석
    project.explore_data()
    
    # 3. 데이터 시각화
    project.visualize_data()
    
    # 4. 데이터 전처리
    project.preprocess_data('구매여부')
    
    # 5. 모델 학습
    project.train_model('classification')
    
    # 6. 모델 평가
    project.evaluate_model()
    
    # 7. 보고서 생성
    project.generate_report()
    
    print("🎉 프로젝트 완료!")
    return project

# 팀 프로젝트 가이드라인
def print_project_guidelines():
    """팀 프로젝트 가이드라인 출력"""
    
    guidelines = """
    
🎯 팀 프로젝트 가이드라인
{"="*50}

📋 프로젝트 진행 단계:

1️⃣ 주제 선정 (1일)
   - 팀원들과 관심 분야 논의
   - 데이터 가용성 확인
   - 실현 가능한 목표 설정

2️⃣ 데이터 수집 (2-3일)
   - 공공데이터 포털 활용
   - 웹 크롤링 (필요시)
   - 데이터 품질 검증

3️⃣ 데이터 분석 (3-4일)
   - EDA 수행
   - 전처리 및 정제
   - 시각화를 통한 인사이트 도출

4️⃣ 모델링 (2-3일)
   - 적절한 알고리즘 선택
   - 모델 학습 및 평가
   - 성능 개선

5️⃣ 결과 정리 (1-2일)
   - 보고서 작성
   - 발표 자료 준비
   - 코드 정리 및 문서화

🔧 권장 도구:
   - Jupyter Notebook (분석)
   - GitHub (협업)
   - Streamlit (웹앱)
   - Canva (발표자료)

📊 추천 프로젝트 주제:

🏠 부동산 분야:
   - 지역별 부동산 가격 예측
   - 전세/매매 가격 비교 분석

📱 소셜미디어:
   - 트위터 감성 분석
   - 유튜브 댓글 분석

🛒 이커머스:
   - 고객 구매 패턴 분석
   - 상품 추천 시스템

🌤️ 기상/환경:
   - 날씨 데이터 기반 판매량 예측
   - 미세먼지 농도 분석

⚽ 스포츠:
   - 스포츠 경기 결과 예측
   - 선수 성과 분석

💡 성공 팁:
   1. 명확한 문제 정의
   2. 단계별 목표 설정
   3. 정기적인 팀 미팅
   4. 코드 리뷰 문화
   5. 결과 해석의 중요성

📈 평가 기준:
   - 기술적 완성도 (40%)
   - 창의성 및 독창성 (30%)
   - 발표 및 소통 (20%)
   - 팀워크 (10%)
    """
    
    print(guidelines)

if __name__ == "__main__":
    print("📚 Week 09-10: 팀 프로젝트 템플릿")
    print()
    
    choice = input("1. 샘플 프로젝트 실행  2. 가이드라인 보기  선택: ")
    
    if choice == "1":
        run_sample_project()
    else:
        print_project_guidelines()
