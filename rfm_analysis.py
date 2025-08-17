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

print("=" * 80)
print("   업계 표준 RFM 분석 - UK 온라인 리테일")
print("   5분위 기반 점수 & 8-11개 표준 세그먼트")
print("=" * 80)

# ========================================================================
# 1. 데이터 로드 및 기본 전처리
# ========================================================================
print("\n📊 1단계: 데이터 로드 및 기본 전처리")
print("-" * 50)

# 데이터 로드
df = pd.read_csv('data/processed/cleaned_retail_data.csv', encoding='ISO-8859-1')
print(f"원본 데이터 크기: {df.shape}")

# 날짜 형식 변환
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 도메인 지식: 정상 거래만 필터링 (환불/취소 제외, 양수 값만)
df_clean = df[
    (df['Quantity'] > 0) & 
    (df['UnitPrice'] > 0) & 
    (~df['InvoiceNo'].astype(str).str.contains('C', na=False)) &
    (df['CustomerID'].notna())
].copy()

print(f"정상 거래 데이터: {df_clean.shape}")
print(f"고유 고객 수: {df_clean['CustomerID'].nunique():,}명")

# 총 구매액 계산
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

# 데이터 기간 확인
print(f"\n데이터 기간:")
print(f"- 시작일: {df_clean['InvoiceDate'].min()}")
print(f"- 종료일: {df_clean['InvoiceDate'].max()}")
print(f"- 총 기간: {(df_clean['InvoiceDate'].max() - df_clean['InvoiceDate'].min()).days}일")

# ========================================================================
# 2. RFM 메트릭 계산
# ========================================================================
print("\n📈 2단계: RFM 메트릭 계산")
print("-" * 50)

# 분석 기준일 설정 (데이터의 마지막 날짜 + 1일)
analysis_date = df_clean['InvoiceDate'].max() + timedelta(days=1)
print(f"분석 기준일: {analysis_date}")

# 고객별 RFM 계산
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency (주문 횟수)
    'TotalAmount': 'sum'  # Monetary
}).reset_index()

# 컬럼명 변경
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"RFM 데이터 생성 완료: {rfm.shape}")
print("\nRFM 기본 통계:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))

# ========================================================================
# 3. 5분위 기반 RFM 점수 계산 (업계 표준)
# ========================================================================
print("\n🎯 3단계: 5분위(Quintile) 기반 RFM 점수 계산")
print("-" * 50)

# 도메인 지식: 5분위 기반 점수 (1-5점)
# - Recency: 낮을수록 좋음 (최근 구매) → 역순 라벨
# - Frequency: 높을수록 좋음 → 정순 라벨  
# - Monetary: 높을수록 좋음 → 정순 라벨

# Recency 점수 (1=오래됨, 5=최근)
rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 
                         q=5, 
                         labels=[5, 4, 3, 2, 1])

# Frequency 점수 (1=낮음, 5=높음)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 
                         q=5, 
                         labels=[1, 2, 3, 4, 5])

# Monetary 점수 (1=낮음, 5=높음)
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 
                         q=5, 
                         labels=[1, 2, 3, 4, 5])

# 점수를 정수형으로 변환
rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)

# RFM 종합 점수 (3자리 숫자로 표현)
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + \
                   rfm['F_Score'].astype(str) + \
                   rfm['M_Score'].astype(str)

print("RFM 점수 분포:")
print(f"- R_Score: {rfm['R_Score'].value_counts().sort_index().to_dict()}")
print(f"- F_Score: {rfm['F_Score'].value_counts().sort_index().to_dict()}")
print(f"- M_Score: {rfm['M_Score'].value_counts().sort_index().to_dict()}")

# ========================================================================
# 4. 업계 표준 8-11개 세그먼트 정의
# ========================================================================
print("\n🏷️ 4단계: 업계 표준 세그먼트 정의 (8-11개)")
print("-" * 50)

def assign_rfm_segment(row):
    """
    도메인 지식: 업계 표준 8-11개 세그먼트
    참고: Klaviyo, Adobe Analytics, HubSpot 등 주요 플랫폼 표준
    """
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    
    # 1. Champions/VIP (최고 가치 고객)
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    
    # 2. Loyal Customers (충성 고객)
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    
    # 3. Potential Loyalists (잠재 충성 고객)
    elif r >= 3 and f >= 2 and m >= 2:
        return 'Potential Loyalists'
    
    # 4. New Customers (신규 고객)
    elif r >= 4 and f == 1:
        return 'New Customers'
    
    # 5. Promising (유망 고객)
    elif r >= 3 and f == 1 and m >= 3:
        return 'Promising'
    
    # 6. Need Attention (관심 필요)
    elif r == 3 and f >= 2 and m >= 2:
        return 'Need Attention'
    
    # 7. About to Sleep (수면 임박)
    elif r == 2 and f >= 2:
        return 'About to Sleep'
    
    # 8. At Risk (위험 고객)
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    
    # 9. Cannot Lose Them (중요 이탈 위험)
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Cannot Lose Them'
    
    # 10. Hibernating (동면 고객)
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Hibernating'
    
    # 11. Lost (이탈 고객)
    elif r == 1:
        return 'Lost'
    
    # 기타
    else:
        return 'Others'

# 세그먼트 할당
rfm['Segment'] = rfm.apply(assign_rfm_segment, axis=1)

# 세그먼트 분포 확인
segment_dist = rfm['Segment'].value_counts()
print("\n세그먼트 분포:")
for segment, count in segment_dist.items():
    percentage = (count / len(rfm)) * 100
    print(f"{segment:20s}: {count:5d}명 ({percentage:5.1f}%)")

# ========================================================================
# 5. 이탈 정의 및 라벨링 (120일 기준)
# ========================================================================
print("\n⚠️ 5단계: 이탈 정의 (120일 기준)")
print("-" * 50)

# 도메인 지식: 일반 온라인 리테일 재구매 주기 고려, 120일(4개월) 기준
CHURN_THRESHOLD_DAYS = 120

rfm['is_churned'] = (rfm['Recency'] > CHURN_THRESHOLD_DAYS).astype(int)

print(f"이탈 기준: 마지막 구매 후 {CHURN_THRESHOLD_DAYS}일 초과")
print(f"전체 이탈률: {rfm['is_churned'].mean():.1%}")
print(f"- 활성 고객: {(~rfm['is_churned'].astype(bool)).sum():,}명")
print(f"- 이탈 고객: {rfm['is_churned'].sum():,}명")

# 세그먼트별 이탈률
segment_churn = rfm.groupby('Segment').agg({
    'is_churned': 'mean',
    'CustomerID': 'count'
}).round(3)
segment_churn.columns = ['Churn_Rate', 'Customer_Count']
segment_churn = segment_churn.sort_values('Churn_Rate', ascending=False)

print("\n세그먼트별 이탈률:")
print(segment_churn)

# ========================================================================
# 6. 고가치 고객 정의 (상위 20%)
# ========================================================================
print("\n💎 6단계: 고가치 고객 정의 (Monetary 상위 20%)")
print("-" * 50)

# 도메인 지식: Pareto 원칙에 따라 상위 20% 고객이 매출의 60-80% 기여
monetary_80th_percentile = rfm['Monetary'].quantile(0.8)
rfm['is_high_value'] = (rfm['Monetary'] >= monetary_80th_percentile).astype(int)

high_value_customers = rfm[rfm['is_high_value'] == 1]
print(f"고가치 고객 기준: £{monetary_80th_percentile:.2f} 이상")
print(f"고가치 고객 수: {len(high_value_customers):,}명 ({len(high_value_customers)/len(rfm)*100:.1f}%)")
print(f"고가치 고객 매출 기여도: {high_value_customers['Monetary'].sum()/rfm['Monetary'].sum()*100:.1f}%")

# ========================================================================
# 7. 추가 비즈니스 메트릭 계산
# ========================================================================
print("\n📊 7단계: 추가 비즈니스 메트릭 계산")
print("-" * 50)

# 평균 주문 가치 (AOV)
rfm['avg_order_value'] = rfm['Monetary'] / rfm['Frequency']

# 구매 주기 (일 단위) - Frequency가 2 이상인 고객만
# 고객 생애 기간을 Frequency로 나눔
customer_lifetime = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (x.max() - x.min()).days
})['InvoiceDate']

rfm = rfm.merge(customer_lifetime.rename('customer_lifetime_days'), 
                left_on='CustomerID', right_index=True, how='left')

rfm['avg_purchase_interval'] = rfm.apply(
    lambda x: x['customer_lifetime_days'] / (x['Frequency'] - 1) if x['Frequency'] > 1 else np.nan,
    axis=1
)

print("비즈니스 메트릭 요약:")
print(f"- 평균 주문 가치(AOV): £{rfm['avg_order_value'].mean():.2f}")
print(f"- 중간값 주문 가치: £{rfm['avg_order_value'].median():.2f}")
print(f"- 평균 구매 주기: {rfm['avg_purchase_interval'].mean():.1f}일 (2회 이상 구매 고객)")

# ========================================================================
# 8. 시계열 트렌드 분석을 위한 피처 준비
# ========================================================================
print("\n📈 8단계: 시계열 트렌드 분석 피처 생성")
print("-" * 50)

# 고객별 구매 이력 시계열 데이터 생성
def calculate_customer_trends(customer_id, transactions_df):
    """
    도메인 지식: 구매 빈도/금액의 시간적 변화 추적
    """
    customer_trans = transactions_df[transactions_df['CustomerID'] == customer_id].sort_values('InvoiceDate')
    
    if len(customer_trans) < 2:
        return pd.Series({
            'frequency_trend': 0,
            'monetary_trend': 0,
            'interval_trend': 0
        })
    
    # 시간을 반으로 나누어 전반기/후반기 비교
    mid_date = customer_trans['InvoiceDate'].min() + \
               (customer_trans['InvoiceDate'].max() - customer_trans['InvoiceDate'].min()) / 2
    
    first_half = customer_trans[customer_trans['InvoiceDate'] <= mid_date]
    second_half = customer_trans[customer_trans['InvoiceDate'] > mid_date]
    
    # 빈도 트렌드 (후반기 빈도 / 전반기 빈도 - 1)
    freq_trend = 0
    if len(first_half) > 0:
        freq_trend = (len(second_half) / len(first_half)) - 1
    
    # 금액 트렌드
    monetary_trend = 0
    if first_half['TotalAmount'].sum() > 0:
        monetary_trend = (second_half['TotalAmount'].sum() / first_half['TotalAmount'].sum()) - 1
    
    # 구매 간격 트렌드 (간격이 늘어나면 양수)
    purchase_dates = customer_trans.groupby(customer_trans['InvoiceDate'].dt.date)['InvoiceDate'].first()
    if len(purchase_dates) > 2:
        intervals = np.diff(purchase_dates.values).astype('timedelta64[D]').astype(int)
        if len(intervals) > 1:
            # 선형 회귀의 기울기로 트렌드 계산
            interval_trend = np.polyfit(range(len(intervals)), intervals, 1)[0]
        else:
            interval_trend = 0
    else:
        interval_trend = 0
    
    return pd.Series({
        'frequency_trend': freq_trend,
        'monetary_trend': monetary_trend,
        'interval_trend': interval_trend
    })

# 샘플링하여 트렌드 계산 (전체 계산은 시간이 오래 걸림)
print("고객 트렌드 계산 중... (샘플링)")
sample_customers = rfm[rfm['Frequency'] >= 2]['CustomerID'].sample(min(1000, len(rfm)))
trends = sample_customers.apply(lambda x: calculate_customer_trends(x, df_clean))

# 트렌드 요약
if len(trends) > 0:
    print(f"\n트렌드 분석 결과 (샘플 {len(trends)}명):")
    print(f"- 구매 빈도 감소 고객: {(trends['frequency_trend'] < -0.3).sum()}명")
    print(f"- 구매 금액 감소 고객: {(trends['monetary_trend'] < -0.3).sum()}명")
    print(f"- 구매 간격 증가 고객: {(trends['interval_trend'] > 10).sum()}명")

# ========================================================================
# 9. 세그먼트별 전략 및 특성 분석
# ========================================================================
print("\n🎯 9단계: 세그먼트별 특성 및 마케팅 전략")
print("-" * 50)

segment_analysis = rfm.groupby('Segment').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'sum'],
    'avg_order_value': 'mean',
    'is_churned': 'mean',
    'is_high_value': 'sum',
    'CustomerID': 'count'
}).round(2)

segment_analysis.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 
                           'Total_Revenue', 'Avg_AOV', 'Churn_Rate', 
                           'High_Value_Count', 'Customer_Count']

# 매출 기여도 추가
segment_analysis['Revenue_Contribution'] = (segment_analysis['Total_Revenue'] / 
                                           rfm['Monetary'].sum() * 100).round(1)

print("\n세그먼트별 상세 분석:")
print(segment_analysis[['Customer_Count', 'Revenue_Contribution', 'Avg_AOV', 
                        'Churn_Rate', 'High_Value_Count']])

# 주요 세그먼트별 전략 제안
strategies = {
    'Champions': '독점 혜택, VIP 프로그램, 조기 접근권 제공',
    'Loyal Customers': '로열티 프로그램 강화, 추천 보상 제공',
    'Potential Loyalists': '참여 유도 캠페인, 브랜드 가치 전달',
    'New Customers': '온보딩 프로그램, 두 번째 구매 유도 할인',
    'At Risk': '재활성화 캠페인, 개인화된 오퍼 제공',
    'Cannot Lose Them': '긴급 개입, 1:1 커뮤니케이션, 특별 할인',
    'Lost': 'Win-back 캠페인, 대폭 할인, 제품 업데이트 알림'
}

print("\n📋 세그먼트별 마케팅 전략:")
for segment, strategy in strategies.items():
    if segment in rfm['Segment'].values:
        count = len(rfm[rfm['Segment'] == segment])
        print(f"\n{segment} ({count}명):")
        print(f"  → {strategy}")

# ========================================================================
# 10. 시각화
# ========================================================================
print("\n📊 10단계: 시각화 생성")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. RFM 점수 분포
ax = axes[0, 0]
rfm_scores = pd.DataFrame({
    'Recency': rfm['R_Score'].value_counts().sort_index(),
    'Frequency': rfm['F_Score'].value_counts().sort_index(),
    'Monetary': rfm['M_Score'].value_counts().sort_index()
})
rfm_scores.plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_title('RFM Score Distribution (1-5 Quintiles)', fontsize=12, fontweight='bold')
ax.set_xlabel('Score')
ax.set_ylabel('Number of Customers')
ax.legend(title='Metric')
ax.grid(True, alpha=0.3)

# 2. 세그먼트 분포
ax = axes[0, 1]
segment_counts = rfm['Segment'].value_counts().head(10)
colors = plt.cm.Set3(np.linspace(0, 1, len(segment_counts)))
segment_counts.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Customer Segment Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Number of Customers')
ax.set_ylabel('Segment')

# 3. 세그먼트별 매출 기여도
ax = axes[0, 2]
revenue_by_segment = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False).head(10)
revenue_pct = (revenue_by_segment / rfm['Monetary'].sum() * 100)
revenue_pct.plot(kind='bar', ax=ax, color='gold')
ax.set_title('Revenue Contribution by Segment (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Segment')
ax.set_ylabel('Revenue Contribution (%)')
ax.tick_params(axis='x', rotation=45)

# 4. Recency vs Frequency 산점도
ax = axes[1, 0]
scatter = ax.scatter(rfm['Recency'], rfm['Frequency'], 
                    c=rfm['Monetary'], cmap='YlOrRd', 
                    alpha=0.6, s=20)
ax.set_title('Recency vs Frequency (colored by Monetary)', fontsize=12, fontweight='bold')
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency (orders)')
ax.axvline(x=CHURN_THRESHOLD_DAYS, color='red', linestyle='--', alpha=0.5, label='Churn Threshold')
ax.legend()
plt.colorbar(scatter, ax=ax, label='Monetary (£)')

# 5. 이탈률 by 세그먼트
ax = axes[1, 1]
churn_by_segment = rfm.groupby('Segment')['is_churned'].mean().sort_values(ascending=False).head(10)
bars = ax.bar(range(len(churn_by_segment)), churn_by_segment.values, 
              color=['red' if x > 0.5 else 'orange' if x > 0.3 else 'green' 
                     for x in churn_by_segment.values])
ax.set_xticks(range(len(churn_by_segment)))
ax.set_xticklabels(churn_by_segment.index, rotation=45, ha='right')
ax.set_title('Churn Rate by Segment', fontsize=12, fontweight='bold')
ax.set_ylabel('Churn Rate')
ax.axhline(y=rfm['is_churned'].mean(), color='black', linestyle='--', alpha=0.5, 
          label=f'Avg: {rfm["is_churned"].mean():.1%}')
ax.legend()

# 6. 고가치 고객 분포
ax = axes[1, 2]
high_value_dist = rfm.groupby('Segment')['is_high_value'].sum().sort_values(ascending=False).head(10)
high_value_dist.plot(kind='bar', ax=ax, color='purple')
ax.set_title('High-Value Customers by Segment', fontsize=12, fontweight='bold')
ax.set_xlabel('Segment')
ax.set_ylabel('Number of High-Value Customers')
ax.tick_params(axis='x', rotation=45)

plt.suptitle('RFM Analysis Dashboard - Industry Standard Approach', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('rfm_analysis_standard.png', dpi=300, bbox_inches='tight')
plt.show()

# ========================================================================
# 11. 결과 저장
# ========================================================================
print("\n💾 11단계: 결과 저장")
print("-" * 50)

# RFM 데이터 저장
rfm.to_csv('rfm_analysis_results.csv', index=False)
print("✅ RFM 분석 결과 저장: rfm_analysis_results.csv")

# 세그먼트 요약 저장
segment_summary = segment_analysis.copy()
segment_summary.to_csv('segment_summary.csv')
print("✅ 세그먼트 요약 저장: segment_summary.csv")

# 고위험 고객 리스트 (At Risk + Cannot Lose Them)
high_risk_segments = ['At Risk', 'Cannot Lose Them']
high_risk_customers = rfm[rfm['Segment'].isin(high_risk_segments)].sort_values('Monetary', ascending=False)
high_risk_customers.to_csv('high_risk_customers.csv', index=False)
print(f"✅ 고위험 고객 리스트 저장: high_risk_customers.csv ({len(high_risk_customers)}명)")

# ========================================================================
# 12. 최종 요약
# ========================================================================
print("\n" + "=" * 80)
print("🎯 RFM 분석 완료 - 업계 표준 적용")
print("=" * 80)

print("\n📊 전체 요약:")
print(f"- 총 고객 수: {len(rfm):,}명")
print(f"- 평균 Recency: {rfm['Recency'].mean():.1f}일")
print(f"- 평균 Frequency: {rfm['Frequency'].mean():.1f}회")
print(f"- 평균 Monetary: £{rfm['Monetary'].mean():.2f}")
print(f"- 전체 이탈률: {rfm['is_churned'].mean():.1%}")

print("\n💎 핵심 인사이트:")
top3_segments = segment_analysis.nlargest(3, 'Revenue_Contribution')
print("매출 기여도 TOP 3 세그먼트:")
for idx, (segment, row) in enumerate(top3_segments.iterrows(), 1):
    print(f"  {idx}. {segment}: {row['Revenue_Contribution']:.1f}% (고객 {row['Customer_Count']:.0f}명)")

print("\n⚠️ 주의 필요 세그먼트:")
risk_segments = rfm[rfm['Segment'].isin(['At Risk', 'Cannot Lose Them'])].groupby('Segment').agg({
    'CustomerID': 'count',
    'Monetary': 'sum'
})
for segment, row in risk_segments.iterrows():
    print(f"  - {segment}: {row['CustomerID']}명, 잠재 손실 매출 £{row['Monetary']:,.0f}")

print("\n✨ 분석 완료! 다음 단계는 시계열 피처 추출 및 예측 모델링입니다.")
print("=" * 80)