import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('Online_Retail.csv')

print("=== 1. 데이터 전처리 시작 ===")

# 1. 기본 정보 확인
print(f"원본 데이터 크기: {df.shape}")
print(f"CustomerID 결측치: {df['CustomerID'].isnull().sum():,}개")
print(f"고유 고객 수: {df['CustomerID'].nunique():,}명")

# 2. CustomerID가 없는 거래 제거
print("\n--- CustomerID 결측치 제거 ---")
df_clean = df.dropna(subset=['CustomerID']).copy()
print(f"정리 후 데이터 크기: {df_clean.shape}")
print(f"제거된 거래: {df.shape[0] - df_clean.shape[0]:,}개")

# 3. InvoiceDate를 datetime으로 변환
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# 4. 데이터 기간 확인
print(f"\n--- 데이터 기간 ---")
print(f"시작일: {df_clean['InvoiceDate'].min()}")
print(f"종료일: {df_clean['InvoiceDate'].max()}")
print(f"총 기간: {(df_clean['InvoiceDate'].max() - df_clean['InvoiceDate'].min()).days}일")

# 5. 이상치 분석 및 처리
print(f"\n--- 이상치 분석 ---")
print(f"음수 수량 거래: {(df_clean['Quantity'] < 0).sum():,}개")
print(f"0 이하 단가 거래: {(df_clean['UnitPrice'] <= 0).sum():,}개")
print(f"환불 Invoice (C로 시작): {df_clean['InvoiceNo'].str.contains('C', na=False).sum():,}개")

# 6. 환불 데이터 분석
print(f"\n--- 환불 데이터 상세 분석 ---")
returns = df_clean[df_clean['InvoiceNo'].str.contains('C', na=False)]
print(f"환불 거래 수: {len(returns):,}")
print(f"환불 관련 고객 수: {returns['CustomerID'].nunique():,}")

# 환불 데이터 예시
if len(returns) > 0:
    print("\n환불 데이터 예시:")
    print(returns[['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID']].head())

# 7. 정상 거래만 필터링 (분석 목적에 따라 선택)
print(f"\n--- 정상 거래 필터링 ---")
df_positive = df_clean[
    (df_clean['Quantity'] > 0) & 
    (df_clean['UnitPrice'] > 0) &
    (~df_clean['InvoiceNo'].str.contains('C', na=False))
].copy()

print(f"정상 거래 데이터 크기: {df_positive.shape}")
print(f"정상 거래 고객 수: {df_positive['CustomerID'].nunique():,}명")

# 8. 총 구매액 계산
df_positive['TotalAmount'] = df_positive['Quantity'] * df_positive['UnitPrice']

# 9. 기본 통계 확인
print(f"\n--- 정상 거래 기본 통계 ---")
print(df_positive[['Quantity', 'UnitPrice', 'TotalAmount']].describe())

# 10. 국가별 분포 확인
print(f"\n--- 국가별 거래 분포 (상위 10개국) ---")
country_dist = df_positive['Country'].value_counts().head(10)
print(country_dist)

# 11. 영국 데이터만 사용할지 결정 (선택사항)
uk_ratio = (df_positive['Country'] == 'United Kingdom').mean()
print(f"\n영국 거래 비율: {uk_ratio:.1%}")

# 영국 데이터만 사용하는 경우
df_uk = df_positive[df_positive['Country'] == 'United Kingdom'].copy()
print(f"영국 데이터 크기: {df_uk.shape}")
print(f"영국 고객 수: {df_uk['CustomerID'].nunique():,}명")

# 12. 최종 데이터 저장을 위한 준비
print(f"\n--- 전처리 완료 ---")
print("사용 가능한 데이터셋:")
print(f"1. df_positive: 전체 정상 거래 ({df_positive.shape[0]:,}개)")
print(f"2. df_uk: 영국 정상 거래 ({df_uk.shape[0]:,}개)")

# 어떤 데이터를 사용할지 선택
use_uk_only = input("\n영국 데이터만 사용하시겠습니까? (y/n): ").lower().strip()

if use_uk_only == 'y':
    df_final = df_uk.copy()
    print("✅ 영국 데이터만 사용합니다.")
else:
    df_final = df_positive.copy()
    print("✅ 전체 국가 데이터를 사용합니다.")

print(f"\n최종 분석 데이터: {df_final.shape}")
print(f"분석 대상 고객 수: {df_final['CustomerID'].nunique():,}명")

# 데이터 저장 (선택사항)
df_final.to_csv('cleaned_retail_data.csv', index=False)
print("\n다음 단계: RFM 분석을 진행합니다.")