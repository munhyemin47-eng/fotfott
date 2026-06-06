import streamlit as st
import tempfile
import os
import trimesh
import plotly.graph_objects as go
from processor import analyze_video

st.set_page_config(page_title="AI 3D 발 측정기", layout="centered")
st.title("👟 AI 3D 발 사이즈 측정기")

uploaded_file = st.file_uploader("영상 파일 (.mp4) 업로드", type=["mp4"])

if uploaded_file is not None:
    if st.button("측정 시작"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        try:
            with st.spinner('AI가 딥러닝 분석 중입니다... 잠시만 기다려주세요!'):
                # 1. 영상 분석
                length, width, instep = analyze_video(tfile.name)
            
            st.success("✅ 분석 완료!")
            
            # 2. 결과 수치 출력
            col1, col2, col3 = st.columns(3)
            col1.metric("발 길이", f"{length:.1f} mm")
            col2.metric("발 볼", f"{width:.1f} mm")
            col3.metric("발등 높이", f"{instep:.1f} mm")
            
            # 3. 3D 모델 렌더링 (측정 시작 버튼 안으로 이동)
            base_path = os.path.dirname(os.path.abspath(__file__))
            obj_path = os.path.join(base_path, '11536_foot_V3.obj')
            
            if os.path.exists(obj_path):
                mesh = trimesh.load(obj_path)
                # 수치 기반 스케일 조정
                mesh.apply_scale([length/260.0, width/100.0, instep/60.0])
                
                fig = go.Figure(data=[go.Mesh3d(
                    x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                    color='cyan', opacity=0.8
                )])
                st.plotly_chart(fig)
            else:
                st.error("3D 모델 파일(11536_foot_V3.obj)을 찾을 수 없습니다.")

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
        finally:
            if os.path.exists(tfile.name):
                os.remove(tfile.name)