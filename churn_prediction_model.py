import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 모델링 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report, 
                           confusion_matrix, precision_recall_curve, f1_score,
                           precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import pickle

# XGBoost와 LightGBM 가용성 확인
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")

print("=" * 80)
print("   통합 고객 이탈 예측 모델")
print("   RFM + 시계열 피처 + 모델 앙상블")
print("=" * 80)

# ========================================================================
# 1. 데이터 로드 및 통합
# ========================================================================
print("\n📊 1단계: 데이터 로드 및 통합")
print("-" * 50)

# RFM 분석 결과 로드
try:
    rfm = pd.read_csv('results/rfm_result/rfm_analysis_results.csv')
    print(f"✅ RFM 데이터 로드: {rfm.shape}")
except FileNotFoundError:
    print("❌ rfm_analysis_results.csv 파일을 찾을 수 없습니다.")
    print("   먼저 rfm_analysis.py를 실행하세요.")
    exit()

# 원본 거래 데이터 로드 (시계열 피처 생성용)
try:
    df = pd.read_csv('data/processed/cleaned_retail_data.csv', encoding='ISO-8859-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    print(f"✅ 거래 데이터 로드: {df.shape}")
except FileNotFoundError:
    # 대체 경로 시도
    try:
        df = pd.read_csv('data/processed/cleaned_retail_data.csv', encoding='ISO-8859-1')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        print(f"✅ 거래 데이터 로드 (대체 경로): {df.shape}")
    except:
        print("❌ 거래 데이터를 찾을 수 없습니다.")
        exit()

# 정상 거래만 필터링
df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (~df['InvoiceNo'].str.contains('C', na=False)) &
    (df['CustomerID'].notna())
].copy()
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# 분석 기준일
analysis_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
print(f"분석 기준일: {analysis_date}")

# ========================================================================
# 2. 시계열 피처 추출
# ========================================================================
print("\n⏰ 2단계: 시계열 피처 추출")
print("-" * 50)

def extract_time_features(customer_id, transactions_df, analysis_date):
    """
    도메인 지식: 구매 패턴의 시간적 변화를 포착하는 피처
    """
    customer_trans = transactions_df[transactions_df['CustomerID'] == customer_id].sort_values('InvoiceDate')
    
    features = {}
    
    # 기본 통계
    features['total_transactions'] = len(customer_trans)
    features['unique_purchase_days'] = customer_trans['InvoiceDate'].dt.date.nunique()
    features['unique_products'] = customer_trans['StockCode'].nunique()
    
    if len(customer_trans) < 2:
        # 단일 구매 고객
        features['avg_days_between_purchases'] = 999  # 큰 값으로 설정
        features['std_days_between_purchases'] = 0
        features['purchase_interval_trend'] = 0
        features['monetary_trend'] = 0
        features['frequency_trend'] = 0
        features['purchase_regularity'] = 0
        for days in [30, 60, 90]:
            features[f'recent_activity_ratio_{days}d'] = 0
            features[f'recent_monetary_ratio_{days}d'] = 0
        return features
    
    # 구매 간격 분석
    purchase_dates = pd.to_datetime(customer_trans.groupby(
        customer_trans['InvoiceDate'].dt.date)['InvoiceDate'].first())
    
    if len(purchase_dates) > 1:
        intervals = np.diff(purchase_dates.values).astype('timedelta64[D]').astype(float)
        features['avg_days_between_purchases'] = np.mean(intervals)
        features['std_days_between_purchases'] = np.std(intervals)
        
        # 구매 간격 트렌드 (증가하면 양수)
        if len(intervals) > 1:
            features['purchase_interval_trend'] = np.polyfit(range(len(intervals)), intervals, 1)[0]
        else:
            features['purchase_interval_trend'] = 0
            
        # 구매 규칙성 (변동계수의 역수)
        if features['avg_days_between_purchases'] > 0:
            cv = features['std_days_between_purchases'] / features['avg_days_between_purchases']
            features['purchase_regularity'] = 1 / (1 + cv)
        else:
            features['purchase_regularity'] = 0
    else:
        features['avg_days_between_purchases'] = 999
        features['std_days_between_purchases'] = 0
        features['purchase_interval_trend'] = 0
        features['purchase_regularity'] = 0
    
    # 시간대별 분석 (전반기 vs 후반기)
    time_span = (customer_trans['InvoiceDate'].max() - customer_trans['InvoiceDate'].min()).days
    if time_span > 0:
        mid_date = customer_trans['InvoiceDate'].min() + timedelta(days=time_span/2)
        
        first_half = customer_trans[customer_trans['InvoiceDate'] <= mid_date]
        second_half = customer_trans[customer_trans['InvoiceDate'] > mid_date]
        
        # 금액 트렌드
        if first_half['TotalAmount'].sum() > 0:
            features['monetary_trend'] = (second_half['TotalAmount'].sum() / 
                                         first_half['TotalAmount'].sum()) - 1
        else:
            features['monetary_trend'] = 0
            
        # 빈도 트렌드
        if len(first_half) > 0:
            features['frequency_trend'] = (len(second_half) / len(first_half)) - 1
        else:
            features['frequency_trend'] = 0
    else:
        features['monetary_trend'] = 0
        features['frequency_trend'] = 0
    
    # 최근 활동 비율
    for days in [30, 60, 90]:
        recent_date = analysis_date - timedelta(days=days)
        recent_trans = customer_trans[customer_trans['InvoiceDate'] >= recent_date]
        
        features[f'recent_activity_ratio_{days}d'] = len(recent_trans) / len(customer_trans)
        
        if customer_trans['TotalAmount'].sum() > 0:
            features[f'recent_monetary_ratio_{days}d'] = (recent_trans['TotalAmount'].sum() / 
                                                          customer_trans['TotalAmount'].sum())
        else:
            features[f'recent_monetary_ratio_{days}d'] = 0
    
    return features

# 배치 처리로 시계열 피처 추출
print("시계열 피처 추출 중...")
time_features_list = []
batch_size = 500
customer_ids = rfm['CustomerID'].values

for i in range(0, len(customer_ids), batch_size):
    batch_ids = customer_ids[i:i+batch_size]
    
    for customer_id in batch_ids:
        features = extract_time_features(customer_id, df_clean, analysis_date)
        features['CustomerID'] = customer_id
        time_features_list.append(features)
    
    if (i + batch_size) % 2000 == 0:
        print(f"  처리 완료: {min(i + batch_size, len(customer_ids))}/{len(customer_ids)}")

time_features_df = pd.DataFrame(time_features_list)
print(f"✅ 시계열 피처 생성 완료: {time_features_df.shape}")

# RFM과 시계열 피처 통합
data = rfm.merge(time_features_df, on='CustomerID', how='left')
print(f"✅ 통합 데이터: {data.shape}")

# ========================================================================
# 3. 추가 피처 엔지니어링
# ========================================================================
print("\n🔧 3단계: 추가 피처 엔지니어링")
print("-" * 50)

# 도메인 지식: RFM 상호작용 피처 (Recency 제외 - 리키지 방지)
data['FM_Score'] = data['F_Score'] * data['M_Score']
data['F_Score_squared'] = data['F_Score'] ** 2
data['M_Score_squared'] = data['M_Score'] ** 2

# 이탈 위험 지표 생성
# 도메인 지식: 구매 패턴 악화 신호
data['interval_risk'] = (data['purchase_interval_trend'] > 10).astype(int)  # 간격 10일 이상 증가
data['activity_risk'] = (data['recent_activity_ratio_30d'] < 0.1).astype(int)  # 최근 30일 활동 10% 미만
data['monetary_risk'] = (data['monetary_trend'] < -0.3).astype(int)  # 금액 30% 이상 감소
data['frequency_risk'] = (data['frequency_trend'] < -0.3).astype(int)  # 빈도 30% 이상 감소

# 종합 위험 점수
data['risk_score'] = (data['interval_risk'] + data['activity_risk'] + 
                      data['monetary_risk'] + data['frequency_risk'])

# 세그먼트 위험도 인코딩
# 도메인 지식: 세그먼트별 이탈 위험도 반영
segment_risk_mapping = {
    'Champions': 1,
    'Loyal Customers': 2,
    'Potential Loyalists': 3,
    'Promising': 3,
    'New Customers': 4,
    'Need Attention': 5,
    'About to Sleep': 6,
    'At Risk': 7,
    'Cannot Lose Them': 8,
    'Hibernating': 8,
    'Lost': 9,
    'Others': 5
}
data['segment_risk_level'] = data['Segment'].map(segment_risk_mapping).fillna(5)

# 고객 가치 지표
data['customer_value_score'] = data['F_Score'] * 0.3 + data['M_Score'] * 0.7  # 금액에 더 가중치
data['avg_order_value'] = data['Monetary'] / (data['Frequency'] + 1)  # 0으로 나누기 방지

# 생애 가치 관련
data['lifetime_days'] = data.apply(
    lambda x: (analysis_date - df_clean[df_clean['CustomerID'] == x['CustomerID']]['InvoiceDate'].min()).days 
    if pd.notna(x['CustomerID']) else 0, axis=1
)
data['lifetime_value_per_day'] = data['Monetary'] / (data['lifetime_days'] + 1)

print(f"✅ 피처 엔지니어링 완료: 총 {len(data.columns)}개 피처")

# ========================================================================
# 4. 이탈 라벨 확인 (120일 기준)
# ========================================================================
print("\n🎯 4단계: 이탈 라벨 확인")
print("-" * 50)

# 도메인 지식: 120일(4개월) 이탈 기준
CHURN_THRESHOLD = 120

# RFM 분석에서 이미 정의된 is_churned 확인
if 'is_churned' not in data.columns:
    data['is_churned'] = (data['Recency'] > CHURN_THRESHOLD).astype(int)

print(f"이탈 기준: {CHURN_THRESHOLD}일")
print(f"전체 이탈률: {data['is_churned'].mean():.1%}")
print(f"- 활성 고객: {(data['is_churned'] == 0).sum():,}명")
print(f"- 이탈 고객: {(data['is_churned'] == 1).sum():,}명")
print(f"클래스 비율: 1:{(data['is_churned'] == 0).sum() / max((data['is_churned'] == 1).sum(), 1):.1f}")

# ========================================================================
# 5. 피처 선택 (리키지 방지)
# ========================================================================
print("\n📝 5단계: 피처 선택 (리키지 방지)")
print("-" * 50)

# 도메인 지식: Recency 및 관련 피처 제외 (타겟 리키지 방지)
feature_cols = [
    # RFM 기본 (Recency 제외)
    'Frequency', 'Monetary',
    'F_Score', 'M_Score',
    
    # RFM 파생
    'FM_Score', 'F_Score_squared', 'M_Score_squared',
    'customer_value_score', 'avg_order_value',
    
    # 시계열 피처
    'total_transactions', 'unique_purchase_days', 'unique_products',
    'avg_days_between_purchases', 'std_days_between_purchases',
    'purchase_interval_trend', 'monetary_trend', 'frequency_trend',
    'purchase_regularity',
    
    # 최근 활동
    'recent_activity_ratio_30d', 'recent_activity_ratio_60d', 'recent_activity_ratio_90d',
    'recent_monetary_ratio_30d', 'recent_monetary_ratio_60d', 'recent_monetary_ratio_90d',
    
    # 위험 지표
    'interval_risk', 'activity_risk', 'monetary_risk', 'frequency_risk', 'risk_score',
    
    # 세그먼트 및 가치
    'segment_risk_level', 'is_high_value',
    'lifetime_value_per_day'
]

# 사용 가능한 피처만 선택
available_features = [col for col in feature_cols if col in data.columns]
print(f"사용 가능한 피처 수: {len(available_features)}")

# 결측값 및 무한대 처리
data[available_features] = data[available_features].fillna(0)
data[available_features] = data[available_features].replace([np.inf, -np.inf], 0)
# ========================================================================
# 6. 데이터 분할
# ========================================================================
print("\n📊 6단계: 데이터 분할")
print("-" * 50)

# 피처와 타겟 분리
X = data[available_features]
y = data['is_churned']

# 전체 데이터를 훈련 세트와 임시 세트(검증 + 테스트)로 분할
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 임시 세트를 검증 세트와 테스트 세트로 분할
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"학습 세트: {X_train.shape[0]}개 (이탈률: {y_train.mean():.1%})")
print(f"검증 세트: {X_val.shape[0]}개 (이탈률: {y_val.mean():.1%})")
print(f"테스트 세트: {X_test.shape[0]}개 (이탈률: {y_test.mean():.1%})")

# 데이터 스케일링 추가
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ========================================================================
# 7. 클래스 불균형 처리
# ========================================================================
print("\n⚖️ 7단계: 클래스 불균형 처리")
print("-" * 50)

# SMOTE 적용 (훈련 데이터만)
if y_train.mean() < 0.4:  # 이탈률이 40% 미만일 때만
    print("SMOTE 적용 중...")
    smote = SMOTE(random_state=42, sampling_strategy=0.5)  # 50%까지만 증가
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"균형화 후 훈련 세트: {X_train_balanced.shape[0]}개 (이탈률: {y_train_balanced.mean():.1%})")
else:
    X_train_balanced = X_train_scaled
    y_train_balanced = y_train

# ========================================================================
# 8. 개별 모델 학습
# ========================================================================
print("\n🤖 8단계: 개별 모델 학습")
print("-" * 50)

models = {}
model_scores = {}

# 1. Logistic Regression
print("\n[1/5] Logistic Regression 학습 중...")
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='liblinear'
)
lr_model.fit(X_train_balanced, y_train_balanced)
lr_val_pred = lr_model.predict_proba(X_val_scaled)[:, 1]
lr_val_score = roc_auc_score(y_val, lr_val_pred)
models['LogisticRegression'] = lr_model
model_scores['LogisticRegression'] = lr_val_score
print(f"  검증 AUC: {lr_val_score:.4f}")

# 2. Random Forest
print("\n[2/5] Random Forest 학습 중...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # 원본 데이터 사용
rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
rf_val_score = roc_auc_score(y_val, rf_val_pred)
models['RandomForest'] = rf_model
model_scores['RandomForest'] = rf_val_score
print(f"  검증 AUC: {rf_val_score:.4f}")

# 3. Gradient Boosting
print("\n[3/5] Gradient Boosting 학습 중...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_val_pred = gb_model.predict_proba(X_val)[:, 1]
gb_val_score = roc_auc_score(y_val, gb_val_pred)
models['GradientBoosting'] = gb_model
model_scores['GradientBoosting'] = gb_val_score
print(f"  검증 AUC: {gb_val_score:.4f}")

# 4. XGBoost (가능한 경우)
if XGB_AVAILABLE:
    print("\n[4/5] XGBoost 학습 중...")
    # 클래스 불균형 가중치
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
    xgb_val_score = roc_auc_score(y_val, xgb_val_pred)
    models['XGBoost'] = xgb_model
    model_scores['XGBoost'] = xgb_val_score
    print(f"  검증 AUC: {xgb_val_score:.4f}")

# 5. LightGBM (가능한 경우)
if LGB_AVAILABLE:
    print("\n[5/5] LightGBM 학습 중...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_val_score = roc_auc_score(y_val, lgb_val_pred)
    models['LightGBM'] = lgb_model
    model_scores['LightGBM'] = lgb_val_score
    print(f"  검증 AUC: {lgb_val_score:.4f}")

# ========================================================================
# 9. 모델 앙상블
# ========================================================================
print("\n🎭 9단계: 모델 앙상블")
print("-" * 50)

# 상위 3개 모델 선택
top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
print("\n상위 3개 모델:")
for model_name, score in top_models:
    print(f"  - {model_name}: AUC {score:.4f}")

# Soft Voting 앙상블
print("\n앙상블 모델 생성 중...")
ensemble_estimators = []
for model_name, _ in top_models:
    if model_name == 'LogisticRegression':
        # 스케일링된 데이터를 위한 파이프라인 필요
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', models[model_name])
        ])
        ensemble_estimators.append((model_name, pipeline))
    else:
        ensemble_estimators.append((model_name, models[model_name]))

ensemble_model = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft',
    n_jobs=-1
)

# 앙상블 모델 학습
ensemble_model.fit(X_train, y_train)
ensemble_val_pred = ensemble_model.predict_proba(X_val)[:, 1]
ensemble_val_score = roc_auc_score(y_val, ensemble_val_pred)
print(f"\n앙상블 모델 검증 AUC: {ensemble_val_score:.4f}")

# ========================================================================
# 10. 최종 모델 선택 및 평가
# ========================================================================
print("\n🏆 10단계: 최종 모델 선택 및 평가")
print("-" * 50)

# 모든 모델 비교 (앙상블 포함)
all_scores = model_scores.copy()
all_scores['Ensemble'] = ensemble_val_score

# 최고 성능 모델 선택
best_model_name = max(all_scores, key=all_scores.get)
best_score = all_scores[best_model_name]

print(f"\n최고 성능 모델: {best_model_name} (검증 AUC: {best_score:.4f})")

# 최종 모델로 테스트 세트 평가
if best_model_name == 'Ensemble':
    final_model = ensemble_model
    test_pred = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]
elif best_model_name == 'LogisticRegression':
    final_model = models[best_model_name]
    test_pred = final_model.predict(X_test_scaled)
    test_proba = final_model.predict_proba(X_test_scaled)[:, 1]
else:
    final_model = models[best_model_name]
    test_pred = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]

# 테스트 세트 성능 평가
test_auc = roc_auc_score(y_test, test_proba)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\n📊 테스트 세트 성능:")
print(f"  - AUC: {test_auc:.4f}")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall: {test_recall:.4f}")
print(f"  - F1-Score: {test_f1:.4f}")

# 혼동 행렬
cm = confusion_matrix(y_test, test_pred)
print("\n혼동 행렬:")
print(f"{'':10s} {'예측: 활성':>12s} {'예측: 이탈':>12s}")
print(f"{'실제: 활성':10s} {cm[0,0]:12d} {cm[0,1]:12d}")
print(f"{'실제: 이탈':10s} {cm[1,0]:12d} {cm[1,1]:12d}")

# ========================================================================
# 11. 교차 검증으로 안정성 확인
# ========================================================================
print("\n🔍 11단계: 교차 검증으로 모델 안정성 확인")
print("-" * 50)

# 전체 데이터로 교차 검증
X_all = data[available_features]
y_all = data['is_churned']

cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5-Fold 교차 검증 중...")
if best_model_name == 'LogisticRegression':
    cv_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', models[best_model_name])
    ])
    cv_scores = cross_val_score(cv_pipeline, X_all, y_all, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
elif best_model_name == 'Ensemble':
    cv_scores = cross_val_score(ensemble_model, X_all, y_all, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
else:
    cv_scores = cross_val_score(models[best_model_name], X_all, y_all, cv=cv_folds, scoring='roc_auc', n_jobs=-1)

print(f"\n{best_model_name} 교차 검증 결과:")
print(f"  - 평균 AUC: {cv_scores.mean():.4f}")
print(f"  - 표준편차: {cv_scores.std():.4f}")
print(f"  - 최소 AUC: {cv_scores.min():.4f}")
print(f"  - 최대 AUC: {cv_scores.max():.4f}")

# ========================================================================
# 12. 피처 중요도 분석
# ========================================================================
print("\n📈 12단계: 피처 중요도 분석")
print("-" * 50)

# Tree 기반 모델의 피처 중요도
if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': models[best_model_name].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n상위 15개 중요 피처:")
    print(feature_importance.head(15).to_string(index=False))
    
elif best_model_name == 'LogisticRegression':
    # 로지스틱 회귀 계수
    feature_coef = pd.DataFrame({
        'Feature': available_features,
        'Coefficient': models[best_model_name].coef_[0],
        'Abs_Coefficient': np.abs(models[best_model_name].coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\n상위 15개 중요 피처 (절댓값 기준):")
    print(feature_coef[['Feature', 'Coefficient']].head(15).to_string(index=False))
    
elif best_model_name == 'Ensemble':
    print("앙상블 모델은 개별 피처 중요도를 제공하지 않습니다.")
    # 대신 개별 모델들의 평균 중요도 계산
    importance_sum = np.zeros(len(available_features))
    importance_count = 0
    
    for model_name, _ in top_models:
        if model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
            importance_sum += models[model_name].feature_importances_
            importance_count += 1
    
    if importance_count > 0:
        avg_importance = importance_sum / importance_count
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Avg_Importance': avg_importance
        }).sort_values('Avg_Importance', ascending=False)
        
        print("\n앙상블 구성 모델들의 평균 피처 중요도 (상위 15개):")
        print(feature_importance.head(15).to_string(index=False))

# ========================================================================
# 13. 전체 데이터 예측 및 위험 등급 분류
# ========================================================================
print("\n🎯 13단계: 전체 고객 이탈 확률 예측")
print("-" * 50)

# 전체 데이터에 대한 예측
if best_model_name == 'LogisticRegression':
    X_all_scaled = scaler.transform(X_all)
    all_proba = models[best_model_name].predict_proba(X_all_scaled)[:, 1]
elif best_model_name == 'Ensemble':
    all_proba = ensemble_model.predict_proba(X_all)[:, 1]
else:
    all_proba = models[best_model_name].predict_proba(X_all)[:, 1]

data['churn_probability'] = all_proba

# 위험 등급 분류
data['risk_level'] = pd.cut(data['churn_probability'], 
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High', 'Very High'])

risk_summary = data.groupby('risk_level').agg({
    'CustomerID': 'count',
    'Monetary': ['mean', 'sum'],
    'Frequency': 'mean',
    'is_churned': 'mean',
    'churn_probability': 'mean'
}).round(2)

print("\n위험 등급별 고객 분포:")
print(risk_summary)

# ========================================================================
# 14. 비즈니스 인사이트
# ========================================================================
print("\n💼 14단계: 비즈니스 인사이트")
print("-" * 50)

# 1. 고위험 고객 식별
high_risk_threshold = 0.7
high_risk_customers = data[data['churn_probability'] >= high_risk_threshold].copy()
high_risk_customers = high_risk_customers.sort_values('churn_probability', ascending=False)

print(f"\n🔴 고위험 고객 (이탈 확률 ≥ {high_risk_threshold:.0%}):")
print(f"  - 고객 수: {len(high_risk_customers):,}명")
print(f"  - 평균 Frequency: {high_risk_customers['Frequency'].mean():.1f}")
print(f"  - 평균 Monetary: £{high_risk_customers['Monetary'].mean():.2f}")
print(f"  - 잠재 손실 매출: £{high_risk_customers['Monetary'].sum():,.2f}")

# 2. 세그먼트별 평균 이탈 확률
segment_risk = data.groupby('Segment').agg({
    'churn_probability': ['mean', 'std'],
    'CustomerID': 'count',
    'Monetary': 'sum'
}).round(3)
segment_risk.columns = ['_'.join(col).strip() for col in segment_risk.columns]
segment_risk = segment_risk.sort_values('churn_probability_mean', ascending=False)

print("\n📊 세그먼트별 이탈 위험도:")
print(segment_risk.head(10))

# 3. 조기 경고 신호 고객
early_warning = data[
    (data['purchase_interval_trend'] > 20) & 
    (data['recent_activity_ratio_30d'] < 0.2) &
    (data['is_churned'] == 0) &
    (data['churn_probability'] >= 0.5)
]

print(f"\n⚠️ 조기 경고 신호 고객:")
print(f"  - 대상: {len(early_warning):,}명")
print(f"  - 평균 이탈 확률: {early_warning['churn_probability'].mean():.1%}")
print(f"  - 위험 매출: £{early_warning['Monetary'].sum():,.2f}")

# ========================================================================
# 15. ROI 기반 개입 전략
# ========================================================================
print("\n💰 15단계: ROI 기반 개입 전략")
print("-" * 50)

# 도메인 지식: 재참여 캠페인 비용 및 효과 가정
campaign_cost_per_customer = 5  # £5 per customer
reactivation_success_rate = 0.25  # 25% 성공률
avg_reactivation_value = data[data['is_churned'] == 0]['avg_order_value'].mean()

print(f"\n캠페인 가정:")
print(f"  - 고객당 비용: £{campaign_cost_per_customer}")
print(f"  - 예상 성공률: {reactivation_success_rate:.0%}")
print(f"  - 재활성화 평균 가치: £{avg_reactivation_value:.2f}")

for threshold in [0.5, 0.6, 0.7, 0.8]:
    target_customers = data[
        (data['churn_probability'] >= threshold) & 
        (data['is_churned'] == 0)
    ]
    
    if len(target_customers) > 0:
        expected_cost = len(target_customers) * campaign_cost_per_customer
        expected_reactivations = len(target_customers) * reactivation_success_rate
        expected_revenue = expected_reactivations * avg_reactivation_value
        expected_profit = expected_revenue - expected_cost
        expected_roi = (expected_profit / expected_cost * 100) if expected_cost > 0 else 0
        
        print(f"\n임계값 {threshold:.0%}:")
        print(f"  - 대상 고객: {len(target_customers):,}명")
        print(f"  - 예상 비용: £{expected_cost:,.0f}")
        print(f"  - 예상 재활성화: {expected_reactivations:.0f}명")
        print(f"  - 예상 수익: £{expected_revenue:,.0f}")
        print(f"  - 예상 이익: £{expected_profit:,.0f}")
        print(f"  - 예상 ROI: {expected_roi:.1f}%")

# 최적 임계값 찾기
best_roi = -np.inf
best_threshold = 0.5

for threshold in np.arange(0.4, 0.9, 0.05):
    target_customers = data[
        (data['churn_probability'] >= threshold) & 
        (data['is_churned'] == 0)
    ]
    
    if len(target_customers) > 10:  # 최소 10명 이상
        expected_cost = len(target_customers) * campaign_cost_per_customer
        expected_reactivations = len(target_customers) * reactivation_success_rate
        expected_revenue = expected_reactivations * avg_reactivation_value
        expected_profit = expected_revenue - expected_cost
        roi = (expected_profit / expected_cost * 100) if expected_cost > 0 else 0
        
        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold

print(f"\n🎯 최적 타겟팅 임계값: {best_threshold:.2f} (예상 ROI: {best_roi:.1f}%)")

# ========================================================================
# 16. 시각화
# ========================================================================
print("\n📊 16단계: 시각화 생성")
print("-" * 50)

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 1. 모델 성능 비교
ax = axes[0, 0]
models_list = list(all_scores.keys())
scores_list = list(all_scores.values())
colors = ['skyblue' if m != best_model_name else 'gold' for m in models_list]
bars = ax.bar(range(len(models_list)), scores_list, color=colors)
ax.set_xticks(range(len(models_list)))
ax.set_xticklabels(models_list, rotation=45, ha='right')
ax.set_ylabel('Validation AUC')
ax.set_title('Model Performance Comparison', fontweight='bold')
for bar, score in zip(bars, scores_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
           f'{score:.3f}', ha='center', va='bottom')
ax.set_ylim([min(scores_list) * 0.95, max(scores_list) * 1.05])

# 2. ROC 곡선
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, test_proba)
ax.plot(fpr, tpr, label=f'{best_model_name} (AUC={test_auc:.3f})', linewidth=2, color='blue')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Test Set', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Precision-Recall 곡선
ax = axes[0, 2]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, test_proba)
ax.plot(recall_vals, precision_vals, linewidth=2, color='green')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.grid(True, alpha=0.3)
# F1 최적점 표시
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])
ax.scatter(recall_vals[optimal_idx], precision_vals[optimal_idx], 
          color='red', s=100, zorder=5, label=f'Best F1={f1_scores[optimal_idx]:.3f}')
ax.legend()

# 4. 피처 중요도 (상위 10개)
ax = axes[1, 0]
if 'feature_importance' in locals() and not feature_importance.empty:
    top_features = feature_importance.head(10)
    importance_col = 'Importance' if 'Importance' in top_features.columns else 'Avg_Importance'
    ax.barh(range(len(top_features)), top_features[importance_col].values, color='coral')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Feature Importance', fontweight='bold')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'Feature importance\nnot available', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Feature Importance', fontweight='bold')

# 5. 이탈 확률 분포
ax = axes[1, 1]
ax.hist(data[data['is_churned'] == 0]['churn_probability'], bins=30, 
        alpha=0.6, label='Active', color='green', edgecolor='black')
ax.hist(data[data['is_churned'] == 1]['churn_probability'], bins=30, 
        alpha=0.6, label='Churned', color='red', edgecolor='black')
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
ax.axvline(x=best_threshold, color='blue', linestyle='--', alpha=0.5, 
          label=f'Optimal ROI Threshold ({best_threshold:.2f})')
ax.set_xlabel('Churn Probability')
ax.set_ylabel('Number of Customers')
ax.set_title('Churn Probability Distribution', fontweight='bold')
ax.legend()

# 6. 위험 등급별 고객 분포
ax = axes[1, 2]
risk_counts = data['risk_level'].value_counts()
colors_risk = ['green', 'yellow', 'orange', 'red']
bars = ax.bar(range(len(risk_counts)), risk_counts.values, 
              color=[colors_risk[i] for i in range(len(risk_counts))])
ax.set_xticks(range(len(risk_counts)))
ax.set_xticklabels(risk_counts.index)
ax.set_ylabel('Number of Customers')
ax.set_title('Customer Distribution by Risk Level', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}\n({height/len(data)*100:.1f}%)',
           ha='center', va='bottom')

# 7. 세그먼트별 평균 이탈 확률
ax = axes[2, 0]
segment_churn_prob = data.groupby('Segment')['churn_probability'].mean().sort_values(ascending=False).head(10)
colors_seg = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(segment_churn_prob)))
bars = ax.bar(range(len(segment_churn_prob)), segment_churn_prob.values, color=colors_seg)
ax.set_xticks(range(len(segment_churn_prob)))
ax.set_xticklabels(segment_churn_prob.index, rotation=45, ha='right')
ax.set_ylabel('Average Churn Probability')
ax.set_title('Churn Risk by Segment', fontweight='bold')
ax.axhline(y=data['churn_probability'].mean(), color='black', linestyle='--', 
          alpha=0.5, label=f'Overall Avg: {data["churn_probability"].mean():.2f}')
ax.legend()

# 8. 교차 검증 결과
ax = axes[2, 1]
cv_data = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
    'AUC': cv_scores
})
bars = ax.bar(cv_data['Fold'], cv_data['AUC'], color='lightblue', edgecolor='navy')
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
          label=f'Mean: {cv_scores.mean():.4f}')
ax.set_ylabel('AUC Score')
ax.set_title('Cross-Validation Results', fontweight='bold')
ax.set_ylim([cv_scores.min() * 0.95, cv_scores.max() * 1.02])
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
           f'{score:.3f}', ha='center', va='bottom')
ax.legend()

# 9. ROI 분석
ax = axes[2, 2]
thresholds = np.arange(0.4, 0.9, 0.05)
rois = []
customer_counts = []

for threshold in thresholds:
    target = data[(data['churn_probability'] >= threshold) & (data['is_churned'] == 0)]
    if len(target) > 0:
        cost = len(target) * campaign_cost_per_customer
        revenue = len(target) * reactivation_success_rate * avg_reactivation_value
        roi = ((revenue - cost) / cost * 100) if cost > 0 else 0
        rois.append(roi)
        customer_counts.append(len(target))
    else:
        rois.append(0)
        customer_counts.append(0)

ax2 = ax.twinx()
line1 = ax.plot(thresholds, rois, 'b-', linewidth=2, label='ROI %')
bars = ax2.bar(thresholds, customer_counts, alpha=0.3, color='gray', width=0.03, label='Target Customers')
ax.set_xlabel('Probability Threshold')
ax.set_ylabel('ROI (%)', color='b')
ax2.set_ylabel('Number of Customers', color='gray')
ax.set_title('ROI Analysis by Threshold', fontweight='bold')
ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, 
          label=f'Optimal: {best_threshold:.2f}')
ax.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='gray')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.suptitle(f'Churn Prediction Model Analysis - {best_model_name}', 
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('churn_prediction_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 시각화 저장 완료: churn_prediction_comprehensive.png")

# ========================================================================
# 17. 결과 저장
# ========================================================================
print("\n💾 17단계: 결과 저장")
print("-" * 50)

# 1. 예측 결과 저장
prediction_results = data[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary',
                          'is_churned', 'churn_probability', 'risk_level']].copy()
prediction_results = prediction_results.sort_values('churn_probability', ascending=False)
prediction_results.to_csv('churn_predictions_final.csv', index=False)
print("✅ 예측 결과 저장: churn_predictions_final.csv")

# 2. 고위험 고객 리스트
high_risk_export = high_risk_customers[['CustomerID', 'Segment', 'churn_probability',
                                        'Recency', 'Frequency', 'Monetary',
                                        'risk_score', 'recent_activity_ratio_30d']].copy()
high_risk_export.to_csv('high_risk_customers_final.csv', index=False)
print(f"✅ 고위험 고객 리스트: high_risk_customers_final.csv ({len(high_risk_export)}명)")

# 3. 모델 성능 요약
performance_summary = pd.DataFrame({
    'Model': [best_model_name],
    'Validation_AUC': [best_score],
    'Test_AUC': [test_auc],
    'Test_Precision': [test_precision],
    'Test_Recall': [test_recall],
    'Test_F1': [test_f1],
    'CV_Mean_AUC': [cv_scores.mean()],
    'CV_Std_AUC': [cv_scores.std()],
    'Optimal_ROI_Threshold': [best_threshold],
    'Expected_ROI': [best_roi]
})
performance_summary.to_csv('model_performance_final.csv', index=False)
print("✅ 모델 성능 요약: model_performance_final.csv")

# 4. 피처 중요도
if 'feature_importance' in locals() and not feature_importance.empty:
    feature_importance.to_csv('feature_importance_final.csv', index=False)
    print("✅ 피처 중요도: feature_importance_final.csv")

# 5. 세그먼트별 위험도 분석
segment_analysis = data.groupby('Segment').agg({
    'CustomerID': 'count',
    'churn_probability': ['mean', 'std'],
    'is_churned': 'mean',
    'Monetary': ['mean', 'sum'],
    'risk_score': 'mean',
    'recent_activity_ratio_30d': 'mean'
}).round(3)
segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
segment_analysis.to_csv('segment_risk_analysis_final.csv')
print("✅ 세그먼트 위험도 분석: segment_risk_analysis_final.csv")

# 6. 모델 저장 (재사용을 위해)
import pickle
model_package = {
    'model': final_model,
    'scaler': scaler if best_model_name == 'LogisticRegression' else None,
    'features': available_features,
    'model_name': best_model_name,
    'threshold': best_threshold,
    'performance': {
        'auc': test_auc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    }
}
with open('churn_prediction_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✅ 모델 저장: churn_prediction_model.pkl")

# ========================================================================
# 18. 최종 요약
# ========================================================================
print("\n" + "=" * 80)
print("🎯 통합 이탈 예측 모델 구축 완료")
print("=" * 80)

print("\n📊 데이터 요약:")
print(f"  - 총 고객 수: {len(data):,}명")
print(f"  - 이탈률: {data['is_churned'].mean():.1%}")
print(f"  - 사용된 피처 수: {len(available_features)}")

print("\n🏆 모델 성능:")
print(f"  - 최종 모델: {best_model_name}")
print(f"  - 테스트 AUC: {test_auc:.4f}")
print(f"  - 교차 검증 AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall: {test_recall:.4f}")
print(f"  - F1-Score: {test_f1:.4f}")

print("\n💡 핵심 인사이트:")
if 'feature_importance' in locals() and not feature_importance.empty:
    top3_features = feature_importance.head(3)
    print("  가장 중요한 예측 변수:")
    for idx, row in top3_features.iterrows():
        importance_col = 'Importance' if 'Importance' in row else 'Avg_Importance'
        print(f"    {idx+1}. {row['Feature']}: {row[importance_col]:.4f}")

print("\n🚀 비즈니스 가치:")
print(f"  - 고위험 고객: {len(high_risk_customers):,}명")
print(f"  - 잠재 방어 가능 매출: £{high_risk_customers['Monetary'].sum():,.0f}")
print(f"  - 최적 타겟팅 임계값: {best_threshold:.2f}")
print(f"  - 예상 캠페인 ROI: {best_roi:.1f}%")

print("\n📈 권장 다음 단계:")
print("  1. 고위험 고객 대상 맞춤형 재참여 캠페인 실행")
print("  2. A/B 테스트로 개입 전략 효과 검증")
print("  3. 모델 성능 모니터링 대시보드 구축")
print("  4. 실시간 스코어링 API 개발")
print("  5. 정기적 모델 재학습 파이프라인 구축")

print("\n✨ 통합 이탈 예측 모델 완성!")
print("=" * 80)