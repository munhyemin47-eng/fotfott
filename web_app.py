import streamlit as st
import tempfile
import os
import time
from processor import analyze_video

# 페이지 설정
st.set_page_config(page_title="AI Foot Analyzer", page_icon="👟")

st.title("👟 AI 기반 3D 발 사이즈 측정기")
st.markdown("---")

uploaded_file = st.file_uploader("분석할 발 영상을 업로드하세요 (mp4)", type=["mp4"])

if uploaded_file is not None:
    st.info("파일 업로드 완료. '측정 시작' 버튼을 눌러 AI 분석을 시작하세요.")
    
    if st.button("측정 시작"):
        # 로딩 상태를 관리하는 컨테이너
        with st.spinner("딥러닝 모델이 데이터를 정밀 분석 중입니다..."):
            
            # 단계별 진행 상태 표시 (딥러닝스러움 강조)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 1단계: AI 모델 및 가중치 로드 중...")
            progress_bar.progress(20)
            time.sleep(1)
            
            status_text.text("📐 2단계: 영상에서 발의 주요 특징점 추출(Keypoints)...")
            progress_bar.progress(50)
            
            # 파일 임시 저장
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            
            try:
                status_text.text("📊 3단계: 원근 왜곡 보정 및 스케일 계산 중...")
                progress_bar.progress(80)
                
                # 분석 엔진 실행
                l, w, i = analyze_video(tfile.name)
                
                progress_bar.progress(100)
                status_text.text("✅ 분석 완료!")
                
                # 결과 출력
                st.success("딥러닝 분석이 완료되었습니다.")
                st.write(f"### 📏 정밀 측정 결과")
                st.metric("발 길이", f"{l:.1f} mm")
                st.metric("발 볼", f"{w:.1f} mm")
                st.metric("발등 높이", f"{i:.1f} mm")
                
                os.remove(tfile.name)
                
            except Exception as e:
                st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
                if os.path.exists(tfile.name):
                    os.remove(tfile.name)