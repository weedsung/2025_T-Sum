"""
Week 10-12: 구독형 서비스 고객 이탈 예측 프로젝트
최종 프로젝트: 단계별 개발 과정
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Week 10: 프로젝트 기획 및 문제 정의
# =============================================================================

class Week10_ProjectPlanning:
    """
    10주차: 프로젝트 기획 및 문제 정의 단계
    - 비즈니스 문제 분석
    - 데이터 이해 및 탐색
    - 프로젝트 목표 설정
    """
    
    def __init__(self):
        self.project_background = None
        self.business_problem = None
        self.project_goals = None
        self.data_overview = None
        
    def analyze_business_problem(self):
        """비즈니스 문제 분석 및 배경 조사"""
        print("🎯 Week 10: 프로젝트 기획 및 문제 정의")
        print("="*60)
        print()
        
        print("📊 1. 비즈니스 배경 분석")
        print("-" * 30)
        
        self.project_background = {
            "industry_trend": "구독형 비즈니스 모델의 급속한 성장",
            "market_size": "글로벌 구독 경제 규모: 2025년 1조 달러 예상",
            "key_challenge": "고객 이탈률(Churn Rate) 관리의 중요성 증대",
            "business_impact": "신규 고객 획득 비용 vs 기존 고객 유지 비용 (5:1 비율)"
        }
        
        for key, value in self.project_background.items():
            print(f"• {key}: {value}")
        
        print()
        print("🔍 2. 핵심 비즈니스 문제")
        print("-" * 30)
        
        self.business_problem = {
            "primary_question": "어떤 고객이 구독 서비스를 중단할 가능성이 높은가?",
            "secondary_questions": [
                "고객 이탈에 가장 큰 영향을 미치는 요인은 무엇인가?",
                "이탈 위험 고객을 사전에 식별할 수 있는가?",
                "효과적인 고객 유지 전략은 무엇인가?"
            ],
            "business_value": "이탈 예측을 통한 선제적 고객 관리 및 수익성 개선"
        }
        
        print(f"핵심 질문: {self.business_problem['primary_question']}")
        print("\n세부 질문들:")
        for i, question in enumerate(self.business_problem['secondary_questions'], 1):
            print(f"  {i}. {question}")
        print(f"\n비즈니스 가치: {self.business_problem['business_value']}")
        
    def define_project_goals(self):
        """프로젝트 목표 및 성공 지표 정의"""
        print("\n🎯 3. 프로젝트 목표 설정")
        print("-" * 30)
        
        self.project_goals = {
            "primary_objective": "구독형 서비스 고객의 이탈 여부를 85% 이상의 정확도로 예측",
            "secondary_objectives": [
                "이탈에 영향을 미치는 주요 요인 식별 및 순위화",
                "고객 세그먼트별 이탈 패턴 분석",
                "실무진이 활용 가능한 이탈 위험 고객 리스트 제공"
            ],
            "success_metrics": {
                "accuracy": "85% 이상",
                "precision": "80% 이상 (이탈 예측)",
                "recall": "75% 이상 (이탈 고객 탐지)",
                "auc_score": "0.85 이상"
            },
            "deliverables": [
                "데이터 분석 보고서",
                "예측 모델 및 성능 평가",
                "비즈니스 인사이트 및 액션 플랜",
                "대화형 대시보드 (선택사항)"
            ]
        }
        
        print(f"주요 목표: {self.project_goals['primary_objective']}")
        print("\n세부 목표:")
        for i, obj in enumerate(self.project_goals['secondary_objectives'], 1):
            print(f"  {i}. {obj}")
        
        print("\n성공 지표:")
        for metric, target in self.project_goals['success_metrics'].items():
            print(f"  • {metric}: {target}")
        
        print("\n최종 산출물:")
        for i, deliverable in enumerate(self.project_goals['deliverables'], 1):
            print(f"  {i}. {deliverable}")
    
    def initial_data_exploration(self, data_path=None):
        """초기 데이터 탐색 및 이해"""
        print("\n📊 4. 데이터 개요 및 초기 탐색")
        print("-" * 30)
        
        # 샘플 데이터 생성 (실제 프로젝트에서는 실제 데이터 로드)
        if data_path is None:
            self.data = self._create_telco_sample_data()
            print("✅ 텔레콤 고객 이탈 샘플 데이터 생성")
        else:
            self.data = pd.read_csv(data_path)
            print(f"✅ 데이터 로드: {data_path}")
        
        print(f"\n데이터 크기: {self.data.shape[0]:,}행 × {self.data.shape[1]}열")
        
        # 데이터 구조 분석
        print("\n📋 데이터 구조:")
        print(self.data.info())
        
        # 타겟 변수 분포
        print("\n🎯 타겟 변수 (Churn) 분포:")
        churn_counts = self.data['Churn'].value_counts()
        churn_pct = self.data['Churn'].value_counts(normalize=True) * 100
        
        for value, count in churn_counts.items():
            pct = churn_pct[value]
            print(f"  {value}: {count:,}명 ({pct:.1f}%)")
        
        # 기본 통계
        print("\n📈 수치형 변수 기본 통계:")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        print(self.data[numeric_cols].describe())
        
        return self.data
    
    def _create_telco_sample_data(self):
        """텔레콤 고객 이탈 샘플 데이터 생성"""
        np.random.seed(42)
        n_samples = 7043  # 실제 IBM Telco 데이터셋과 유사한 크기
        
        # 고객 기본 정보
        customer_id = [f"CUST_{i:04d}" for i in range(n_samples)]
        gender = np.random.choice(['Male', 'Female'], n_samples)
        senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
        partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48])
        dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
        
        # 서비스 정보
        tenure = np.random.randint(1, 73, n_samples)  # 1-72개월
        phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
        multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10])
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
        
        # 부가 서비스
        online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.28, 0.50, 0.22])
        online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22])
        device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22])
        tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22])
        streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22])
        streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.39, 0.22])
        
        # 계약 정보
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24])
        paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
        payment_method = np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.34, 0.19, 0.22, 0.25])
        
        # 요금 정보
        monthly_charges = np.random.uniform(18.25, 118.75, n_samples)
        total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
        total_charges = np.maximum(total_charges, monthly_charges)  # 최소값 보정
        
        # 이탈 여부 (복합적 요인으로 결정)
        churn_prob = (
            0.3 * (contract == 'Month-to-month') +
            0.2 * (internet_service == 'Fiber optic') +
            0.15 * (payment_method == 'Electronic check') +
            0.1 * (senior_citizen == 1) +
            0.1 * (partner == 'No') +
            0.05 * (tenure < 12) +
            0.1 * (monthly_charges > 80)
        )
        churn_prob = np.clip(churn_prob, 0.05, 0.8)  # 확률 범위 제한
        churn = np.random.binomial(1, churn_prob, n_samples)
        churn = ['Yes' if x == 1 else 'No' for x in churn]
        
        return pd.DataFrame({
            'customerID': customer_id,
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Churn': churn
        })

# =============================================================================
# Week 11: 데이터 분석 및 모델 개발
# =============================================================================

class Week11_DataAnalysisAndModeling:
    """
    11주차: 데이터 분석 및 모델 개발 단계
    - 심화 EDA 및 특성 분석
    - 데이터 전처리 및 특성 엔지니어링
    - 다양한 모델 개발 및 비교
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}
        
    def comprehensive_eda(self):
        """종합적 탐색적 데이터 분석"""
        print("\n🔍 Week 11: 데이터 분석 및 모델 개발")
        print("="*60)
        print()
        
        print("📊 1. 심화 탐색적 데이터 분석 (EDA)")
        print("-" * 40)
        
        # 이탈률 분석
        self._analyze_churn_patterns()
        
        # 특성별 이탈률 분석
        self._analyze_feature_churn_relationship()
        
        # 상관관계 분석
        self._correlation_analysis()
        
    def _analyze_churn_patterns(self):
        """이탈 패턴 분석"""
        print("🎯 이탈 패턴 분석:")
        
        # 전체 이탈률
        overall_churn_rate = (self.data['Churn'] == 'Yes').mean() * 100
        print(f"전체 이탈률: {overall_churn_rate:.1f}%")
        
        # 계약 유형별 이탈률
        print("\n계약 유형별 이탈률:")
        contract_churn = self.data.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        
        for contract, rate in contract_churn.items():
            print(f"  {contract}: {rate:.1f}%")
        
        # 인터넷 서비스별 이탈률
        print("\n인터넷 서비스별 이탈률:")
        internet_churn = self.data.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).sort_values(ascending=False)
        
        for service, rate in internet_churn.items():
            print(f"  {service}: {rate:.1f}%")
    
    def _analyze_feature_churn_relationship(self):
        """특성과 이탈의 관계 분석"""
        print("\n📈 주요 특성별 이탈률 분석:")
        
        categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                              'PhoneService', 'PaperlessBilling', 'PaymentMethod']
        
        for feature in categorical_features:
            if feature in self.data.columns:
                feature_churn = self.data.groupby(feature)['Churn'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).sort_values(ascending=False)
                
                print(f"\n{feature}별 이탈률:")
                for value, rate in feature_churn.items():
                    print(f"  {value}: {rate:.1f}%")
    
    def _correlation_analysis(self):
        """상관관계 분석"""
        print("\n🔗 수치형 변수 상관관계 분석:")
        
        # 수치형 변수만 선택
        numeric_data = self.data.select_dtypes(include=[np.number]).copy()
        
        # Churn을 수치형으로 변환
        numeric_data['Churn_numeric'] = (self.data['Churn'] == 'Yes').astype(int)
        
        # 상관관계 계산
        correlation_with_churn = numeric_data.corr()['Churn_numeric'].abs().sort_values(ascending=False)
        
        print("이탈과의 상관관계 (절댓값 기준):")
        for feature, corr in correlation_with_churn.items():
            if feature != 'Churn_numeric':
                print(f"  {feature}: {corr:.3f}")
    
    def data_preprocessing(self):
        """데이터 전처리 및 특성 엔지니어링"""
        print("\n🔧 2. 데이터 전처리 및 특성 엔지니어링")
        print("-" * 40)
        
        # 데이터 복사
        processed_data = self.data.copy()
        
        # 1. 결측치 처리
        print("결측치 처리:")
        missing_counts = processed_data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"결측치 발견: {missing_counts.sum()}개")
            # TotalCharges가 문자열인 경우 처리
            if 'TotalCharges' in processed_data.columns:
                processed_data['TotalCharges'] = pd.to_numeric(
                    processed_data['TotalCharges'], errors='coerce'
                )
                processed_data['TotalCharges'].fillna(
                    processed_data['TotalCharges'].median(), inplace=True
                )
        else:
            print("결측치 없음 ✅")
        
        # 2. 특성 엔지니어링
        print("\n특성 엔지니어링:")
        
        # 평균 월 요금 계산
        processed_data['AvgMonthlyCharges'] = processed_data['TotalCharges'] / (processed_data['tenure'] + 1)
        
        # 고객 생애 가치 구간화
        processed_data['TenureGroup'] = pd.cut(
            processed_data['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['0-1년', '1-2년', '2-4년', '4년+']
        )
        
        # 요금 구간화
        processed_data['ChargeGroup'] = pd.cut(
            processed_data['MonthlyCharges'],
            bins=[0, 35, 65, 95, 120],
            labels=['저가', '중저가', '중고가', '고가']
        )
        
        # 서비스 이용 개수
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
        processed_data['ServiceCount'] = 0
        for col in service_cols:
            if col in processed_data.columns:
                processed_data['ServiceCount'] += (processed_data[col] == 'Yes').astype(int)
        
        print("생성된 새로운 특성:")
        print("  • AvgMonthlyCharges: 평균 월 요금")
        print("  • TenureGroup: 계약 기간 구간")
        print("  • ChargeGroup: 요금 구간")
        print("  • ServiceCount: 이용 서비스 개수")
        
        # 3. 범주형 변수 인코딩
        print("\n범주형 변수 인코딩:")
        
        # 타겟 변수 분리
        self.y = (processed_data['Churn'] == 'Yes').astype(int)
        
        # 특성 변수 준비
        feature_data = processed_data.drop(['Churn', 'customerID'], axis=1)
        
        # 범주형 변수 인코딩
        categorical_cols = feature_data.select_dtypes(include=['object', 'category']).columns
        
        encoded_data = feature_data.copy()
        for col in categorical_cols:
            if col not in ['TenureGroup', 'ChargeGroup']:  # 이미 처리된 컬럼 제외
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(encoded_data[col].astype(str))
        
        # 구간화된 변수 처리
        if 'TenureGroup' in encoded_data.columns:
            encoded_data['TenureGroup'] = encoded_data['TenureGroup'].cat.codes
        if 'ChargeGroup' in encoded_data.columns:
            encoded_data['ChargeGroup'] = encoded_data['ChargeGroup'].cat.codes
        
        self.X = encoded_data
        
        print(f"최종 특성 개수: {self.X.shape[1]}개")
        print(f"샘플 개수: {self.X.shape[0]}개")
        
        return self.X, self.y
    
    def develop_models(self):
        """다양한 머신러닝 모델 개발"""
        print("\n🤖 3. 머신러닝 모델 개발 및 비교")
        print("-" * 40)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"훈련 데이터: {X_train.shape[0]}개")
        print(f"테스트 데이터: {X_test.shape[0]}개")
        
        # 특성 정규화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 모델 정의
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        print("\n모델 학습 및 평가:")
        
        for name, model in models_to_train.items():
            print(f"\n{name} 학습 중...")
            
            # 모델 학습
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 성능 평가
            accuracy = (y_pred == y_test).mean()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 교차 검증
            if name == 'Logistic Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # 결과 저장
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  정확도: {accuracy:.3f}")
            print(f"  AUC 점수: {auc_score:.3f}")
            print(f"  교차검증 평균: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # 최고 성능 모델 선정
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        print(f"\n🏆 최고 성능 모델: {best_model_name}")
        print(f"   AUC 점수: {self.results[best_model_name]['auc_score']:.3f}")
        
        return self.models, self.results

# =============================================================================
# Week 12: 최종 구현 및 결과 정리
# =============================================================================

class Week12_FinalizationAndResults:
    """
    12주차: 최종 구현 및 결과 정리 단계
    - 모델 성능 최적화
    - 비즈니스 인사이트 도출
    - 최종 보고서 및 대시보드 작성
    """
    
    def __init__(self, models, results, X, y):
        self.models = models
        self.results = results
        self.X = X
        self.y = y
        self.business_insights = {}
        
    def optimize_best_model(self):
        """최고 성능 모델 최적화"""
        print("\n⚡ Week 12: 최종 구현 및 결과 정리")
        print("="*60)
        print()
        
        print("🔧 1. 모델 성능 최적화")
        print("-" * 30)
        
        # 최고 성능 모델 선정
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_model = self.models[best_model_name]
        
        print(f"최적화 대상 모델: {best_model_name}")
        
        # 특성 중요도 분석 (Random Forest 또는 XGBoost인 경우)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n🔍 특성 중요도 Top 10:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.3f}")
            
            self.feature_importance = feature_importance
        
        return best_model_name, best_model
    
    def generate_business_insights(self):
        """비즈니스 인사이트 도출"""
        print("\n💡 2. 비즈니스 인사이트 도출")
        print("-" * 30)
        
        # 고위험 고객 세그먼트 식별
        self._identify_high_risk_segments()
        
        # 이탈 방지 전략 제안
        self._propose_retention_strategies()
        
        # ROI 계산
        self._calculate_business_impact()
    
    def _identify_high_risk_segments(self):
        """고위험 고객 세그먼트 식별"""
        print("🎯 고위험 고객 세그먼트:")
        
        # 원본 데이터에서 분석 (인코딩 전)
        high_risk_segments = [
            "월 단위 계약 고객 (Month-to-month)",
            "Fiber optic 인터넷 서비스 이용 고객",
            "Electronic check 결제 고객",
            "고령 고객 (Senior Citizen)",
            "파트너가 없는 고객",
            "계약 기간 1년 미만 고객"
        ]
        
        for i, segment in enumerate(high_risk_segments, 1):
            print(f"  {i}. {segment}")
        
        self.business_insights['high_risk_segments'] = high_risk_segments
    
    def _propose_retention_strategies(self):
        """이탈 방지 전략 제안"""
        print("\n📋 이탈 방지 전략:")
        
        retention_strategies = {
            "계약 인센티브": [
                "장기 계약 고객 대상 할인 혜택 제공",
                "월 단위 계약에서 연 단위 계약 전환 시 특별 혜택"
            ],
            "서비스 개선": [
                "Fiber optic 서비스 품질 개선 및 기술 지원 강화",
                "부가 서비스 패키지 할인 제공"
            ],
            "결제 편의성": [
                "자동 결제 전환 시 할인 혜택",
                "다양한 결제 옵션 제공"
            ],
            "고객 관리": [
                "신규 고객 온보딩 프로그램 강화",
                "고위험 고객 대상 개인화된 상담 서비스"
            ]
        }
        
        for category, strategies in retention_strategies.items():
            print(f"\n{category}:")
            for strategy in strategies:
                print(f"  • {strategy}")
        
        self.business_insights['retention_strategies'] = retention_strategies
    
    def _calculate_business_impact(self):
        """비즈니스 임팩트 계산"""
        print("\n💰 예상 비즈니스 임팩트:")
        
        # 가정값들
        assumptions = {
            "총 고객 수": 100000,
            "월 평균 수익 (ARPU)": 65,
            "고객 획득 비용 (CAC)": 300,
            "현재 이탈률": 26.5,
            "예측 정확도": 85.0,
            "이탈 방지 성공률": 30.0
        }
        
        # 계산
        current_churn_customers = assumptions["총 고객 수"] * (assumptions["현재 이탈률"] / 100)
        predicted_churn_customers = current_churn_customers * (assumptions["예측 정확도"] / 100)
        retained_customers = predicted_churn_customers * (assumptions["이탈 방지 성공률"] / 100)
        
        # 연간 수익 보존
        annual_revenue_saved = retained_customers * assumptions["월 평균 수익 (ARPU)"] * 12
        
        # 고객 획득 비용 절약
        acquisition_cost_saved = retained_customers * assumptions["고객 획득 비용 (CAC)"]
        
        total_impact = annual_revenue_saved + acquisition_cost_saved
        
        print(f"예측 가능한 이탈 고객: {predicted_churn_customers:,.0f}명")
        print(f"이탈 방지 가능 고객: {retained_customers:,.0f}명")
        print(f"연간 수익 보존: ${annual_revenue_saved:,.0f}")
        print(f"고객 획득 비용 절약: ${acquisition_cost_saved:,.0f}")
        print(f"총 예상 임팩트: ${total_impact:,.0f}")
        
        self.business_insights['financial_impact'] = {
            'retained_customers': retained_customers,
            'annual_revenue_saved': annual_revenue_saved,
            'acquisition_cost_saved': acquisition_cost_saved,
            'total_impact': total_impact
        }
    
    def create_comprehensive_report(self):
        """종합 프로젝트 보고서 생성"""
        print("\n📊 3. 종합 프로젝트 보고서")
        print("-" * 30)
        
        # 최고 성능 모델 정보
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        best_result = self.results[best_model_name]
        
        report = f"""
        
📊 구독형 서비스 고객 이탈 예측 프로젝트 최종 보고서
{"="*80}

🎯 프로젝트 개요
• 목표: 구독형 서비스 고객의 이탈 여부 예측 및 비즈니스 인사이트 도출
• 데이터: 텔레콤 고객 데이터 ({self.X.shape[0]:,}명, {self.X.shape[1]}개 특성)
• 기간: 3주 (기획 → 개발 → 구현)

📈 모델 성능 결과
• 최고 성능 모델: {best_model_name}
• 정확도: {best_result['accuracy']:.1%}
• AUC 점수: {best_result['auc_score']:.3f}
• 교차검증 점수: {best_result['cv_mean']:.3f} (±{best_result['cv_std']:.3f})

🔍 주요 발견사항
1. 계약 유형이 이탈에 가장 큰 영향을 미침
2. 월 단위 계약 고객의 이탈률이 현저히 높음
3. 인터넷 서비스 유형과 결제 방식도 중요한 요인
4. 신규 고객 (계약 기간 1년 미만)의 이탈 위험이 높음

💡 비즈니스 액션 플랜
1. 장기 계약 전환 인센티브 프로그램 도입
2. 고위험 고객 대상 개인화된 리텐션 캠페인
3. 신규 고객 온보딩 프로세스 개선
4. 서비스 품질 개선 (특히 Fiber optic)

💰 예상 비즈니스 임팩트
• 이탈 방지 가능 고객: {self.business_insights['financial_impact']['retained_customers']:,.0f}명
• 연간 수익 보존: ${self.business_insights['financial_impact']['annual_revenue_saved']:,.0f}
• 총 예상 임팩트: ${self.business_insights['financial_impact']['total_impact']:,.0f}

🚀 향후 개선 방안
1. 실시간 이탈 위험 모니터링 시스템 구축
2. 고객 행동 데이터 추가 수집 및 분석
3. A/B 테스트를 통한 리텐션 전략 효과 검증
4. 딥러닝 모델 적용을 통한 성능 개선

📋 기술적 성과
• 데이터 전처리 및 특성 엔지니어링 완료
• 3가지 머신러닝 모델 비교 분석
• 비즈니스 가치 중심의 인사이트 도출
• 실무 적용 가능한 액션 플랜 제시
        """
        
        print(report)
        return report
    
    def create_action_dashboard(self):
        """실무진을 위한 액션 대시보드"""
        print("\n📋 4. 실무진 액션 대시보드")
        print("-" * 30)
        
        dashboard_info = """
        
🎯 고객 이탈 예측 액션 대시보드
{"="*50}

🚨 즉시 조치 필요 (High Priority)
1. 월 단위 계약 + Fiber optic 고객
   → 장기 계약 전환 제안 + 서비스 품질 점검

2. 신규 고객 (가입 6개월 미만)
   → 온보딩 프로그램 참여 유도

3. Electronic check 결제 고객
   → 자동 결제 전환 혜택 제안

⚠️  주의 관찰 필요 (Medium Priority)
1. 고령 고객 (Senior Citizen)
   → 맞춤형 고객 서비스 제공

2. 파트너 없는 고객
   → 가족 플랜 혜택 안내

3. 부가 서비스 미이용 고객
   → 서비스 패키지 할인 제안

📊 주간 모니터링 지표
• 신규 고위험 고객 수
• 리텐션 캠페인 참여율
• 계약 전환율
• 실제 이탈률 vs 예측 이탈률

📞 고객 상담 스크립트
"안녕하세요, [고객명]님. 더 나은 서비스 제공을 위해 
맞춤형 혜택을 준비했습니다..."
        """
        
        print(dashboard_info)
        return dashboard_info

def run_complete_project():
    """전체 프로젝트 실행"""
    print("🚀 구독형 서비스 고객 이탈 예측 프로젝트 시작")
    print("="*80)
    
    # Week 10: 프로젝트 기획
    week10 = Week10_ProjectPlanning()
    week10.analyze_business_problem()
    week10.define_project_goals()
    data = week10.initial_data_exploration()
    
    # Week 11: 데이터 분석 및 모델링
    week11 = Week11_DataAnalysisAndModeling(data)
    week11.comprehensive_eda()
    X, y = week11.data_preprocessing()
    models, results = week11.develop_models()
    
    # Week 12: 최종 구현 및 결과
    week12 = Week12_FinalizationAndResults(models, results, X, y)
    best_model_name, best_model = week12.optimize_best_model()
    week12.generate_business_insights()
    final_report = week12.create_comprehensive_report()
    action_dashboard = week12.create_action_dashboard()
    
    print("\n🎉 프로젝트 완료!")
    print("✅ 모든 단계가 성공적으로 완료되었습니다.")
    
    return {
        'week10': week10,
        'week11': week11,
        'week12': week12,
        'best_model': best_model,
        'final_report': final_report
    }

# 프로젝트 실행 가이드
def print_project_execution_guide():
    """프로젝트 실행 가이드"""
    
    guide = """
    
🎯 구독형 서비스 고객 이탈 예측 프로젝트 실행 가이드
{"="*80}

📅 3주 프로젝트 일정

Week 10: 프로젝트 기획 및 문제 정의 (5일)
├── Day 1-2: 비즈니스 문제 분석 및 배경 조사
├── Day 3: 프로젝트 목표 및 성공 지표 정의
├── Day 4-5: 데이터 수집 및 초기 탐색

Week 11: 데이터 분석 및 모델 개발 (5일)
├── Day 1-2: 심화 EDA 및 특성 분석
├── Day 3: 데이터 전처리 및 특성 엔지니어링
├── Day 4-5: 다양한 모델 개발 및 비교

Week 12: 최종 구현 및 결과 정리 (5일)
├── Day 1-2: 모델 최적화 및 성능 개선
├── Day 3: 비즈니스 인사이트 도출
├── Day 4-5: 최종 보고서 및 발표 준비

🛠️ 필요한 도구 및 라이브러리
• Python 3.8+
• pandas, numpy (데이터 처리)
• matplotlib, seaborn (시각화)
• scikit-learn (머신러닝)
• xgboost (고급 모델)
• jupyter notebook (개발 환경)

📊 데이터셋 정보
• IBM Telco Customer Churn Dataset
• 7,043명의 고객 데이터
• 21개의 특성 변수
• 이탈 여부 (Churn) 타겟 변수

🎯 학습 목표
1. 실제 비즈니스 문제 해결 경험
2. 데이터 분석 전 과정 실습
3. 다양한 머신러닝 모델 비교
4. 비즈니스 인사이트 도출 능력
5. 결과 커뮤니케이션 스킬

💡 성공을 위한 팁
• 비즈니스 관점에서 문제 접근
• 데이터 품질 확인 철저히
• 모델 성능보다 해석 가능성 중시
• 실무진이 이해할 수 있는 결과 제시
• 지속적인 팀 커뮤니케이션
    """
    
    print(guide)

if __name__ == "__main__":
    print("📚 Week 10-12: 구독형 서비스 고객 이탈 예측 프로젝트")
    print()
    
    choice = input("1. 전체 프로젝트 실행  2. 실행 가이드 보기  선택: ")
    
    if choice == "1":
        project_results = run_complete_project()
    else:
        print_project_execution_guide()