import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("   시계열 피처 엔지니어링 & 이탈 예측 모델")
print("   업계 표준 접근법 - 일반 온라인 리테일")
print("=" * 80)

# ========================================================================
# 1. 데이터 로드 및 준비
# ========================================================================
print("\n📊 1단계: 데이터 로드 및 준비")
print("-" * 50)

# RFM 분석 결과 로드
rfm = pd.read_csv('results/rfm_result/rfm_analysis_results.csv')
print(f"RFM 데이터 로드: {rfm.shape}")

# 원본 거래 데이터 로드 (시계열 피처 생성용)
df = pd.read_csv('data/processed/cleaned_retail_data.csv', encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
# 정상 거래만 필터링
df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (~df['InvoiceNo'].str.contains('C', na=False)) &
    (df['CustomerID'].notna())
].copy()

df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
print(f"정상 거래 데이터: {df_clean.shape}")

# 분석 기준일
analysis_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
print(f"분석 기준일: {analysis_date}")

# ========================================================================
# 2. 시계열 피처 엔지니어링
# ========================================================================
print("\n⏰ 2단계: 시계열 피처 엔지니어링")
print("-" * 50)

def extract_time_series_features(customer_id, transactions_df, analysis_date):
    """
    도메인 지식: 구매 패턴의 시간적 변화를 포착하는 일반적인 피처들
    - 구매 간격의 변화
    - 구매 금액의 변화
    - 구매 빈도의 변화
    - 최근 활동 수준
    """
    customer_trans = transactions_df[transactions_df['CustomerID'] == customer_id].sort_values('InvoiceDate')
    
    features = {}
    
    # 기본 정보
    features['total_transactions'] = len(customer_trans)
    features['unique_purchase_days'] = customer_trans['InvoiceDate'].dt.date.nunique()
    
    if len(customer_trans) < 2:
        # 단일 구매 고객의 경우 기본값 설정
        features['avg_days_between_purchases'] = np.nan
        features['std_days_between_purchases'] = np.nan
        features['purchase_interval_trend'] = 0
        features['purchase_amount_trend'] = 0
        features['frequency_trend'] = 0
        features['monetary_velocity'] = 0
        features['purchase_regularity'] = 0
        features['recent_activity_ratio_30d'] = 0
        features['recent_activity_ratio_60d'] = 0
        features['recent_activity_ratio_90d'] = 0
        return features
    
    # 구매 날짜별 집계
    daily_purchases = customer_trans.groupby(customer_trans['InvoiceDate'].dt.date).agg({
        'TotalAmount': 'sum',
        'InvoiceNo': 'nunique'
    }).reset_index()
    daily_purchases.columns = ['Date', 'Amount', 'Orders']
    
    # 1. 구매 간격 분석
    purchase_dates = pd.to_datetime(daily_purchases['Date'])
    intervals = np.diff(purchase_dates).astype('timedelta64[D]').astype(float)
    
    features['avg_days_between_purchases'] = np.mean(intervals) if len(intervals) > 0 else 0
    features['std_days_between_purchases'] = np.std(intervals) if len(intervals) > 0 else 0
    features['min_days_between_purchases'] = np.min(intervals) if len(intervals) > 0 else 0
    features['max_days_between_purchases'] = np.max(intervals) if len(intervals) > 0 else 0
    
    # 구매 간격 트렌드 (간격이 증가하면 양수)
    if len(intervals) > 1:
        features['purchase_interval_trend'] = np.polyfit(range(len(intervals)), intervals, 1)[0]
    else:
        features['purchase_interval_trend'] = 0
    
    # 2. 구매 금액 분석
    amounts = daily_purchases['Amount'].values
    
    # 구매 금액 트렌드
    if len(amounts) > 1:
        features['purchase_amount_trend'] = np.polyfit(range(len(amounts)), amounts, 1)[0]
        
        # 최근 vs 과거 금액 비교
        mid_point = len(amounts) // 2
        recent_avg_amount = np.mean(amounts[mid_point:])
        past_avg_amount = np.mean(amounts[:mid_point])
        features['amount_trend_ratio'] = (recent_avg_amount / (past_avg_amount + 1)) - 1
    else:
        features['purchase_amount_trend'] = 0
        features['amount_trend_ratio'] = 0
    
    # 3. 구매 빈도 분석
    # 시간을 3개 구간으로 나누어 빈도 변화 측정
    time_span = (customer_trans['InvoiceDate'].max() - customer_trans['InvoiceDate'].min()).days
    if time_span > 0:
        third = time_span / 3
        
        period1 = customer_trans[customer_trans['InvoiceDate'] <= 
                                 customer_trans['InvoiceDate'].min() + timedelta(days=third)]
        period2 = customer_trans[(customer_trans['InvoiceDate'] > 
                                  customer_trans['InvoiceDate'].min() + timedelta(days=third)) &
                                 (customer_trans['InvoiceDate'] <= 
                                  customer_trans['InvoiceDate'].min() + timedelta(days=2*third))]
        period3 = customer_trans[customer_trans['InvoiceDate'] > 
                                 customer_trans['InvoiceDate'].min() + timedelta(days=2*third)]
        
        freq1 = len(period1) / max(third, 1)
        freq2 = len(period2) / max(third, 1)
        freq3 = len(period3) / max(third, 1)
        
        # 빈도 트렌드 (감소하면 음수)
        if freq1 > 0:
            features['frequency_trend'] = (freq3 - freq1) / freq1
        else:
            features['frequency_trend'] = 0
            
        # 빈도 가속도 (변화율의 변화)
        features['frequency_acceleration'] = (freq3 - freq2) - (freq2 - freq1)
    else:
        features['frequency_trend'] = 0
        features['frequency_acceleration'] = 0
    
    # 4. 구매 규칙성 (변동계수의 역수)
    if features['avg_days_between_purchases'] > 0:
        cv = features['std_days_between_purchases'] / features['avg_days_between_purchases']
        features['purchase_regularity'] = 1 / (1 + cv)  # 0~1 사이 값, 1에 가까울수록 규칙적
    else:
        features['purchase_regularity'] = 0
    
    # 5. 금액 속도 (Monetary Velocity)
    days_active = (customer_trans['InvoiceDate'].max() - customer_trans['InvoiceDate'].min()).days + 1
    features['monetary_velocity'] = customer_trans['TotalAmount'].sum() / max(days_active, 1)
    
    # 6. 최근 활동 비율 (다양한 기간)
    for days in [30, 60, 90]:
        recent_date = analysis_date - timedelta(days=days)
        recent_trans = customer_trans[customer_trans['InvoiceDate'] >= recent_date]
        features[f'recent_activity_ratio_{days}d'] = len(recent_trans) / len(customer_trans)
        features[f'recent_monetary_ratio_{days}d'] = (recent_trans['TotalAmount'].sum() / 
                                                      customer_trans['TotalAmount'].sum() 
                                                      if len(recent_trans) > 0 else 0)
    
    # 7. 구매 시점 패턴
    features['weekend_purchase_ratio'] = customer_trans['InvoiceDate'].dt.dayofweek.isin([5, 6]).mean()
    features['month_diversity'] = customer_trans['InvoiceDate'].dt.month.nunique() / 12
    
    # 8. 제품 다양성 (카테고리 없이 제품 수로 측정)
    features['unique_products'] = customer_trans['StockCode'].nunique()
    features['avg_products_per_order'] = customer_trans.groupby('InvoiceNo')['StockCode'].nunique().mean()
    
    return features

# 시계열 피처 추출 (전체 고객)
print("시계열 피처 추출 중...")
time_features_list = []

# 배치 처리로 성능 개선
batch_size = 500
customer_ids = rfm['CustomerID'].values

for i in range(0, len(customer_ids), batch_size):
    batch_ids = customer_ids[i:i+batch_size]
    batch_features = []
    
    for customer_id in batch_ids:
        features = extract_time_series_features(customer_id, df_clean, analysis_date)
        features['CustomerID'] = customer_id
        batch_features.append(features)
    
    time_features_list.extend(batch_features)
    
    if (i + batch_size) % 2000 == 0:
        print(f"  처리 완료: {min(i + batch_size, len(customer_ids))}/{len(customer_ids)}")

time_features_df = pd.DataFrame(time_features_list)
print(f"시계열 피처 생성 완료: {time_features_df.shape}")

# RFM 데이터와 병합
data = rfm.merge(time_features_df, on='CustomerID', how='left')
print(f"통합 데이터: {data.shape}")

# ========================================================================
# 3. 추가 피처 엔지니어링
# ========================================================================
print("\n🔧 3단계: 추가 피처 엔지니어링")
print("-" * 50)

# 도메인 지식: 이탈 예측에 유용한 파생 피처들

# 1. RFM 상호작용 피처
data['RF_Score'] = data['R_Score'] * data['F_Score']
data['RM_Score'] = data['R_Score'] * data['M_Score']
data['FM_Score'] = data['F_Score'] * data['M_Score']
data['RFM_Score_Sum'] = data['R_Score'] + data['F_Score'] + data['M_Score']

# 2. 고객 생애 가치 관련
data['customer_lifetime_days'] = data.apply(
    lambda x: (analysis_date - df_clean[df_clean['CustomerID'] == x['CustomerID']]['InvoiceDate'].min()).days if pd.notna(x['CustomerID']) else 0,
    axis=1
)
data['lifetime_value_per_day'] = data['Monetary'] / (data['customer_lifetime_days'] + 1)

# 3. 구매 효율성
data['purchase_efficiency'] = data['Monetary'] / (data['Frequency'] * data['avg_days_between_purchases'] + 1)

# 4. 이탈 위험 지표 (도메인 지식 기반)
# 구매 간격이 평균보다 2배 이상 증가
data['interval_risk'] = (data['purchase_interval_trend'] > data['avg_days_between_purchases']).astype(int)

# 최근 30일 활동이 전체 기간 대비 10% 미만
data['activity_risk'] = (data['recent_activity_ratio_30d'] < 0.1).astype(int)

# 금액 트렌드가 -30% 이하
data['monetary_risk'] = (data['amount_trend_ratio'] < -0.3).astype(int)

# 종합 위험 점수
data['risk_score'] = data['interval_risk'] + data['activity_risk'] + data['monetary_risk']

# 5. 세그먼트 인코딩
segment_mapping = {
    'Champions': 11,
    'Loyal Customers': 10,
    'Potential Loyalists': 9,
    'New Customers': 8,
    'Promising': 7,
    'Need Attention': 6,
    'About to Sleep': 5,
    'At Risk': 4,
    'Cannot Lose Them': 3,
    'Hibernating': 2,
    'Lost': 1,
    'Others': 0
}
data['Segment_Encoded'] = data['Segment'].map(segment_mapping).fillna(0)

print("추가 피처 생성 완료")
print(f"총 피처 수: {len(data.columns)}")

# ========================================================================
# 4. 이탈 라벨 정의 (120일 기준)
# ========================================================================
print("\n🎯 4단계: 이탈 라벨 정의 (120일 기준)")
print("-" * 50)

# 도메인 지식: 일반 온라인 리테일 120일 이탈 기준
CHURN_THRESHOLD = 120
data['is_churned'] = (data['Recency'] > CHURN_THRESHOLD).astype(int)

print(f"이탈 기준: {CHURN_THRESHOLD}일")
print(f"전체 이탈률: {data['is_churned'].mean():.1%}")
print(f"- 활성 고객: {(data['is_churned'] == 0).sum():,}명")
print(f"- 이탈 고객: {(data['is_churned'] == 1).sum():,}명")

# 클래스 불균형 확인
print(f"\n클래스 비율: 1:{(data['is_churned'] == 0).sum() / (data['is_churned'] == 1).sum():.1f}")

# ========================================================================
# 5. 피처 선택 및 전처리
# ========================================================================
print("\n📝 5단계: 피처 선택 및 전처리")
print("-" * 50)

# 예측에 사용할 피처 선택 (Recency 제외 - 타겟 리키지 방지)
feature_cols = [
    # RFM 기본 (Recency 제외)
    'Frequency', 'Monetary',
    'F_Score', 'M_Score',
    
    # RFM 파생
    'FM_Score', 'RFM_Score_Sum',
    'avg_order_value',
    
    # 시계열 피처
    'total_transactions', 'unique_purchase_days',
    'avg_days_between_purchases', 'std_days_between_purchases',
    'min_days_between_purchases', 'max_days_between_purchases',
    'purchase_interval_trend', 'purchase_amount_trend',
    'amount_trend_ratio', 'frequency_trend', 'frequency_acceleration',
    'purchase_regularity', 'monetary_velocity',
    
    # 최근 활동
    'recent_activity_ratio_30d', 'recent_activity_ratio_60d', 'recent_activity_ratio_90d',
    'recent_monetary_ratio_30d', 'recent_monetary_ratio_60d', 'recent_monetary_ratio_90d',
    
    # 구매 패턴
    'weekend_purchase_ratio', 'month_diversity',
    'unique_products', 'avg_products_per_order',
    
    # 위험 지표
    'interval_risk', 'activity_risk', 'monetary_risk', 'risk_score',
    
    # 세그먼트
    'Segment_Encoded',
    
    # 고객 가치
    'is_high_value', 'lifetime_value_per_day', 'purchase_efficiency'
]

# 사용 가능한 피처만 선택
available_features = [col for col in feature_cols if col in data.columns]
print(f"사용 가능한 피처 수: {len(available_features)}")

# 결측값 처리
data[available_features] = data[available_features].fillna(0)

# 무한대 값 처리
data[available_features] = data[available_features].replace([np.inf, -np.inf], 0)

# 피처와 타겟 분리
X = data[available_features]
y = data['is_churned']

print(f"피처 행렬 크기: {X.shape}")
print(f"타겟 분포: {y.value_counts().to_dict()}")

# ========================================================================
# 6. 학습/검증/테스트 데이터 분할 (수정)
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

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# ========================================================================
# 7. 모델 학습 및 평가
# ========================================================================
print("\n🤖 7단계: 모델 학습 및 평가")
print("-" * 50)

models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=20,
        random_state=42
    )
}

results = []
best_model = None
best_auc = 0

for model_name, model in models.items():
    print(f"\n학습 중: {model_name}")
    
    # 모델 학습
    if model_name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred_val = model.predict(X_val_scaled)
        y_proba_val = model.predict_proba(X_val_scaled)[:, 1]
        y_pred_test = model.predict(X_test_scaled)
        y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        y_proba_val = model.predict_proba(X_val)[:, 1]
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # 성능 평가
    val_auc = roc_auc_score(y_val, y_proba_val)
    test_auc = roc_auc_score(y_test, y_proba_test)
    
    # 정밀도, 재현율 계산
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    results.append({
        'Model': model_name,
        'Val_AUC': val_auc,
        'Test_AUC': test_auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    print(f"  검증 AUC: {val_auc:.3f}")
    print(f"  테스트 AUC: {test_auc:.3f}")
    print(f"  정밀도: {precision:.3f}")
    print(f"  재현율: {recall:.3f}")
    print(f"  F1 점수: {f1:.3f}")
    
    if test_auc > best_auc:
        best_auc = test_auc
        best_model = model
        best_model_name = model_name
        best_y_proba = y_proba_test
        best_y_pred = y_pred_test

# 결과 요약
results_df = pd.DataFrame(results)
print("\n📊 모델 성능 비교:")
print(results_df.round(3))

# ========================================================================
# 8. 최적 모델 상세 분석
# ========================================================================
print(f"\n🏆 8단계: 최적 모델 상세 분석 - {best_model_name}")
print("-" * 50)

# 혼동 행렬
cm = confusion_matrix(y_test, best_y_pred)
print("\n혼동 행렬:")
print(f"{'':10s} {'예측: 활성':>12s} {'예측: 이탈':>12s}")
print(f"{'실제: 활성':10s} {cm[0,0]:12d} {cm[0,1]:12d}")
print(f"{'실제: 이탈':10s} {cm[1,0]:12d} {cm[1,1]:12d}")

# 피처 중요도 (Tree 기반 모델의 경우)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n상위 15개 중요 피처:")
    print(feature_importance.head(15).to_string(index=False))

# 임계값 최적화
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, best_y_proba)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold = thresholds[optimal_idx]

print(f"\n최적 임계값: {optimal_threshold:.3f}")
print(f"최적 F1 점수: {f1_scores[optimal_idx]:.3f}")

# ========================================================================
# 9. 비즈니스 인사이트
# ========================================================================
print("\n💼 9단계: 비즈니스 인사이트")
print("-" * 50)

# 전체 데이터에 대한 예측
if best_model_name == 'Logistic Regression':
    full_proba = best_model.predict_proba(scaler.transform(X))[:, 1]
else:
    full_proba = best_model.predict_proba(X)[:, 1]

data['churn_probability'] = full_proba

# 위험 등급 분류
data['risk_level'] = pd.cut(data['churn_probability'], 
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High', 'Very High'])

risk_summary = data.groupby('risk_level').agg({
    'CustomerID': 'count',
    'Monetary': ['mean', 'sum'],
    'Frequency': 'mean',
    'is_churned': 'mean'
}).round(2)

print("\n위험 등급별 고객 분포:")
print(risk_summary)

# 고위험 고객 식별
high_risk_customers = data[data['churn_probability'] >= 0.7]
print(f"\n🔴 고위험 고객 (이탈 확률 ≥ 70%):")
print(f"- 고객 수: {len(high_risk_customers):,}명")
print(f"- 평균 Frequency: {high_risk_customers['Frequency'].mean():.1f}")
print(f"- 평균 Monetary: £{high_risk_customers['Monetary'].mean():.2f}")
print(f"- 잠재 손실 매출: £{high_risk_customers['Monetary'].sum():,.2f}")

# 세그먼트별 평균 이탈 확률
segment_risk = data.groupby('Segment').agg({
    'churn_probability': 'mean',
    'CustomerID': 'count'
}).sort_values('churn_probability', ascending=False)

print("\n세그먼트별 평균 이탈 확률:")
print(segment_risk.head(10).round(3))

# ========================================================================
# 10. 시각화
# ========================================================================
print("\n📊 10단계: 시각화 생성")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. ROC 곡선
from sklearn.metrics import roc_curve
ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, best_y_proba)
ax.plot(fpr, tpr, label=f'{best_model_name} (AUC={best_auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Precision-Recall 곡선
ax = axes[0, 1]
ax.plot(recall_vals[:-1], precision_vals[:-1], linewidth=2)
ax.scatter(recall_vals[optimal_idx], precision_vals[optimal_idx], 
          color='red', s=100, zorder=5, label=f'Optimal (F1={f1_scores[optimal_idx]:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 피처 중요도 (상위 10개)
ax = axes[0, 2]
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['Importance'].values, color='skyblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values)
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Feature Importance', fontweight='bold')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Feature Importance', fontweight='bold')

# 4. 이탈 확률 분포
ax = axes[1, 0]
ax.hist(data[data['is_churned'] == 0]['churn_probability'], bins=30, 
        alpha=0.6, label='Active', color='green', edgecolor='black')
ax.hist(data[data['is_churned'] == 1]['churn_probability'], bins=30, 
        alpha=0.6, label='Churned', color='red', edgecolor='black')
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
ax.axvline(x=optimal_threshold, color='blue', linestyle='--', alpha=0.5, 
          label=f'Optimal Threshold ({optimal_threshold:.2f})')
ax.set_xlabel('Churn Probability')
ax.set_ylabel('Number of Customers')
ax.set_title('Churn Probability Distribution', fontweight='bold')
ax.legend()

# 5. 위험 등급별 고객 분포
ax = axes[1, 1]
risk_counts = data['risk_level'].value_counts()
colors = ['green', 'yellow', 'orange', 'red']
bars = ax.bar(range(len(risk_counts)), risk_counts.values, 
              color=[colors[i] for i in range(len(risk_counts))])
ax.set_xticks(range(len(risk_counts)))
ax.set_xticklabels(risk_counts.index)
ax.set_ylabel('Number of Customers')
ax.set_title('Customer Distribution by Risk Level', fontweight='bold')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height)}\n({height/len(data)*100:.1f}%)',
           ha='center', va='bottom')

# 6. 시계열 피처 상관관계
ax = axes[1, 2]
time_series_features = ['purchase_interval_trend', 'frequency_trend', 
                        'amount_trend_ratio', 'recent_activity_ratio_30d']
time_series_features = [f for f in time_series_features if f in data.columns]
if len(time_series_features) > 0:
    corr_with_churn = data[time_series_features + ['churn_probability']].corr()['churn_probability'][:-1]
    colors = ['red' if x > 0 else 'blue' for x in corr_with_churn.values]
    bars = ax.bar(range(len(corr_with_churn)), corr_with_churn.values, color=colors)
    ax.set_xticks(range(len(corr_with_churn)))
    ax.set_xticklabels(corr_with_churn.index, rotation=45, ha='right')
    ax.set_ylabel('Correlation with Churn Probability')
    ax.set_title('Time Series Features Correlation', fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Churn Prediction Analysis - {best_model_name}', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('churn_prediction_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ========================================================================
# 11. 실행 가능한 인사이트 생성
# ========================================================================
print("\n🎯 11단계: 실행 가능한 인사이트")
print("-" * 50)

# 1. 조기 경고 신호
print("\n⚠️ 조기 경고 신호:")
early_warning = data[
    (data['purchase_interval_trend'] > 20) & 
    (data['recent_activity_ratio_30d'] < 0.2) &
    (data['is_churned'] == 0)
]
print(f"- 구매 간격 증가 + 최근 활동 감소: {len(early_warning):,}명")
print(f"  평균 이탈 확률: {early_warning['churn_probability'].mean():.1%}")

# 2. 세그먼트별 액션 플랜
print("\n📋 세그먼트별 액션 플랜:")

action_plans = {
    'At Risk': {
        'action': '긴급 재참여 캠페인',
        'tactics': ['개인화된 할인 쿠폰', '독점 제품 미리보기', '무료 배송']
    },
    'Cannot Lose Them': {
        'action': '1:1 고객 관리',
        'tactics': ['전담 매니저 배정', 'VIP 혜택', '맞춤형 상품 추천']
    },
    'Hibernating': {
        'action': 'Win-back 캠페인',
        'tactics': ['복귀 할인', '신제품 알림', '브랜드 스토리 공유']
    },
    'About to Sleep': {
        'action': '관심 유도',
        'tactics': ['이메일 빈도 조정', '관심사 기반 콘텐츠', '소셜 미디어 타겟팅']
    }
}

for segment, plan in action_plans.items():
    segment_data = data[data['Segment'] == segment]
    if len(segment_data) > 0:
        print(f"\n{segment} ({len(segment_data):,}명, 평균 이탈 확률: {segment_data['churn_probability'].mean():.1%}):")
        print(f"  → {plan['action']}")
        for tactic in plan['tactics']:
            print(f"    • {tactic}")

# 3. 비용-효과 분석
print("\n💰 비용-효과 분석:")

# 가정: 재참여 캠페인 비용 £5/고객, 성공률 25%, 평균 재활성화 가치 = 평균 AOV
campaign_cost_per_customer = 5
reactivation_success_rate = 0.25
avg_reactivation_value = data['avg_order_value'].mean()

for threshold in [0.5, 0.6, 0.7]:
    target_customers = data[
        (data['churn_probability'] >= threshold) & 
        (data['is_churned'] == 0)
    ]
    
    expected_cost = len(target_customers) * campaign_cost_per_customer
    expected_reactivations = len(target_customers) * reactivation_success_rate
    expected_revenue = expected_reactivations * avg_reactivation_value
    expected_roi = ((expected_revenue - expected_cost) / expected_cost * 100) if expected_cost > 0 else 0
    
    print(f"\n임계값 {threshold:.0%}:")
    print(f"  - 대상 고객: {len(target_customers):,}명")
    print(f"  - 예상 비용: £{expected_cost:,.0f}")
    print(f"  - 예상 재활성화: {expected_reactivations:.0f}명")
    print(f"  - 예상 수익: £{expected_revenue:,.0f}")
    print(f"  - 예상 ROI: {expected_roi:.1f}%")

# ========================================================================
# 12. 결과 저장
# ========================================================================
print("\n💾 12단계: 결과 저장")
print("-" * 50)

# 1. 예측 결과 저장
prediction_results = data[['CustomerID', 'Segment', 'Recency', 'Frequency', 'Monetary',
                          'is_churned', 'churn_probability', 'risk_level']].copy()
prediction_results = prediction_results.sort_values('churn_probability', ascending=False)
prediction_results.to_csv('churn_predictions.csv', index=False)
print("✅ 예측 결과 저장: churn_predictions.csv")

# 2. 고위험 고객 리스트
high_risk_export = high_risk_customers[['CustomerID', 'Segment', 'churn_probability',
                                        'Recency', 'Frequency', 'Monetary',
                                        'recent_activity_ratio_30d',
                                        'purchase_interval_trend']].copy()
high_risk_export.to_csv('high_risk_customers_prediction.csv', index=False)
print(f"✅ 고위험 고객 리스트 저장: high_risk_customers_prediction.csv ({len(high_risk_export)}명)")

# 3. 모델 성능 요약
model_performance = pd.DataFrame(results)
model_performance.to_csv('model_performance.csv', index=False)
print("✅ 모델 성능 저장: model_performance.csv")

# 4. 피처 중요도 (Tree 기반 모델의 경우)
if hasattr(best_model, 'feature_importances_'):
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("✅ 피처 중요도 저장: feature_importance.csv")

# 5. 세그먼트별 위험도 분석
segment_risk_analysis = data.groupby('Segment').agg({
    'CustomerID': 'count',
    'churn_probability': ['mean', 'std'],
    'is_churned': 'mean',
    'Monetary': ['mean', 'sum'],
    'recent_activity_ratio_30d': 'mean',
    'purchase_interval_trend': 'mean'
}).round(3)
segment_risk_analysis.columns = ['_'.join(col).strip() for col in segment_risk_analysis.columns]
segment_risk_analysis.to_csv('segment_risk_analysis.csv')
print("✅ 세그먼트 위험도 분석 저장: segment_risk_analysis.csv")

# ========================================================================
# 13. 최종 요약
# ========================================================================
print("\n" + "=" * 80)
print("🎯 이탈 예측 모델 구축 완료 - 업계 표준 적용")
print("=" * 80)

print("\n📊 데이터 요약:")
print(f"- 총 고객 수: {len(data):,}명")
print(f"- 이탈률: {data['is_churned'].mean():.1%}")
print(f"- 사용된 피처 수: {len(available_features)}")

print("\n🏆 모델 성능:")
print(f"- 최적 모델: {best_model_name}")
print(f"- 테스트 AUC: {best_auc:.3f}")
print(f"- 최적 임계값: {optimal_threshold:.3f}")

print("\n💡 핵심 인사이트:")
if hasattr(best_model, 'feature_importances_'):
    top3_features = feature_importance.head(3)
    print("가장 중요한 예측 변수:")
    for idx, row in top3_features.iterrows():
        print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.3f}")

print("\n🚀 비즈니스 가치:")
print(f"- 고위험 고객: {len(high_risk_customers):,}명")
print(f"- 잠재 방어 가능 매출: £{high_risk_customers['Monetary'].sum():,.0f}")
print(f"- 조기 경고 대상: {len(early_warning):,}명")

print("\n📈 다음 단계:")
print("1. 고위험 고객 대상 재참여 캠페인 실행")
print("2. A/B 테스트를 통한 개입 전략 검증")
print("3. 모델 성능 모니터링 및 정기적 재학습")
print("4. 실시간 스코어링 시스템 구축")

print("\n✨ 시계열 피처 엔지니어링 및 예측 모델 완성!")
print("=" * 80)