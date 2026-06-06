import streamlit as st
import tempfile
import os

st.title("👟 3D 발 사이즈 측정기")

uploaded_file = st.file_uploader("영상 파일 업로드 (mp4)", type=["mp4"])

if uploaded_file is not None:
    st.write("파일이 준비되었습니다.")
    if st.button("측정 시작"):
        try:
            # 1. 파일 임시 저장
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            
            # 2. 분석 엔진 호출
            st.write("분석 엔진을 실행합니다...")
            from processor import analyze_video
            l, w, i = analyze_video(tfile.name)
            
            # 3. 결과 표시
            st.success("분석 완료!")
            st.write(f"### 측정 결과: 길이 {l:.1f}mm, 볼 {w:.1f}mm, 발등 {i:.1f}mm")
            
            os.remove(tfile.name)
        except Exception as e:
            st.error(f"오류 발생: {e}")