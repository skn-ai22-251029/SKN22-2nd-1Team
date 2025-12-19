import streamlit as st
import pandas as pd
import sys
import os

# --- [STEP 1] 경로 설정 (반드시 모든 import보다 위에 와야 함) ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # app 폴더
project_root = os.path.abspath(os.path.join(current_dir, "..")) # 최상위 폴더

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- [STEP 2] 모듈 임포트 ---
# 이제 경로가 설정되었으므로 문제없이 불러올 수 있습니다.
from service.PurchaseIntentService10 import PurchaseIntentService
#  from adapters.model_loader import load_model
# 수정 후 (에러 메시지에 근거한 실제 이름으로 변경)
from adapters.model_loader import JoblibArtifactLoader
from adapters.purchase_intent_pr_auc_adapter import PurchaseIntentPRAUCModelAdapter

# --- [STEP 3] 데이터 및 모델 로드 ---
@st.cache_data
def load_data():
    # 데이터 경로가 프로젝트 루트 기준인지 확인하세요
    return pd.read_csv("data/processed/test.csv")

df = load_data()

# 1차 수정
# model = load_model("artifacts/best_pr_auc_balancedrf.joblib")

# 2차 수정 
# loader = JoblibArtifactLoader()
# model = loader.load("artifacts/best_pr_auc_balancedrf.joblib")

# 3차 수정
# 1. 생성 시점에 모델 경로를 인자로 전달합니다.
model_path = "artifacts/best_pr_auc_balancedrf.joblib"
loader = JoblibArtifactLoader(path=model_path)

# 2. 로더 객체 내부의 로드 메서드를 호출합니다. (메서드 이름이 load가 맞는지 확인 필요)
model = loader.load()
adapter = PurchaseIntentPRAUCModelAdapter(model)
service = PurchaseIntentService(adapter)

# --- [STEP 4] UI 구성 및 로직 ---
st.title("🚨 고위험 이탈 탐지 & 마케팅 액션 추천")

idx = st.selectbox("세션 선택", df.index)
row = df.loc[idx]

X_one = pd.DataFrame([row.drop("Revenue", errors="ignore")])
proba = adapter.predict_proba(X_one).iloc[0]

risk = service.classify_risk(proba)
action = service.recommend_action(row.to_dict(), proba)

st.metric("구매 확률", f"{proba*100:.1f}%")

if risk == "HIGH_RISK":
    st.error("🚨 이탈 위험 높음")
elif risk == "OPPORTUNITY":
    st.warning("⚠️ 전환 가능성 있음")
else:
    st.success("✅ 구매 유력")

st.info(f"📌 추천 액션: {action}")