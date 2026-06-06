import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os, tempfile
from processor import analyze_video

st.title("👟 3D 발 사이즈 측정기")

uploaded_file = st.file_uploader("영상 파일 업로드", type=["mp4"])

if uploaded_file:
    if st.button("측정 시작"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            
            # 분석 엔진 호출
            l, w, i = analyze_video(tfile.name)
            
            # 결과 표시
            st.write(f"### 측정 결과: 길이 {l:.1f}mm, 볼 {w:.1f}mm, 발등 {i:.1f}mm")
            
            # 3D 모델 렌더링
            path = "models/11536_foot_V3.obj"
            v, f = [], []
            with open(path, 'r') as file:
                for line in file:
                    if line.startswith('v '): v.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith('f '): f.append([int(p.split('/')[0])-1 for p in line.split()[1:4]])
            v, f = np.array(v), np.array(f)
            size = np.max(v, axis=0) - np.min(v, axis=0)
            
            # 스케일링
            v[:,0] *= (w/10)/(size[0]+1e-6); v[:,1] *= (l/10)/(size[1]+1e-6); v[:,2] *= (i/10)/(size[2]+1e-6)
            
            fig = go.Figure(data=[go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2], color='peachpuff')])
            st.plotly_chart(fig)
            
            os.remove(tfile.name)