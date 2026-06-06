import streamlit as st
import tempfile
import os

st.title("👟 AI 3D 발 사이즈 측정기")

# 파일 업로드
uploaded_file = st.file_uploader("영상 파일 업로드 (mp4)", type=["mp4"])

if uploaded_file is not None:
    # 측정 시작 버튼
    if st.button("측정 시작"):
        try:
            # 1. 파일 저장
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close() # 닫아줘야 분석 엔진에서 읽을 수 있음
            
            st.write("✅ 파일 저장 완료, 분석 엔진 가동 중...")
            
            # 2. 분석 엔진 호출
            from processor import analyze_video
            l, w, i = analyze_video(tfile.name)
            
            # 3. 결과 표시
            st.success("딥러닝 분석 완료!")
            st.write(f"### 측정 결과")
            st.write(f"길이: {l:.1f}mm / 볼: {w:.1f}mm / 발등: {i:.1f}mm")
            
            # 4. 임시 파일 삭제
            if os.path.exists(tfile.name):
                os.remove(tfile.name)
                
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.write("로그를 확인하세요.")