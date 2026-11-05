from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit
from audio_features import load_audio, windows, features

st.set_page_config(page_title='한민고 생태음향 종 동정', page_icon='🐸', layout='wide')
BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    with np.load(BASE / 'species_model.npz', allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}

st.title('🐸 한민고 생태음향 종 동정')
st.write('개구리·새·매미 등 19개 클래스의 독립적인 탐지 점수를 확인하세요.')
st.warning('점수는 정답 확률이 아니며 합계가 100%일 필요는 없습니다. 음악·잡음에서 오탐이 생길 수 있습니다.')
st.caption('구간 평균은 지속적인 반응을, 순간 최고는 짧은 소리에 대한 반응을 보여줍니다. 긴 녹음의 짧은 울음은 평균이 낮을 수 있습니다.')
upload = st.file_uploader('분석할 음원 선택', type=['wav', 'mp3', 'flac', 'ogg', 'aif', 'aiff'])
st.caption('이 웹 버전은 서버 부담을 줄이기 위해 파일당 30MB, 길이 5분까지 지원합니다. MP4/M4A는 WAV 또는 MP3로 변환해 주세요.')
if upload is not None:
    if upload.size > 30 * 1024 * 1024:
        st.error('30MB 이하의 파일을 올려 주세요.')
        st.stop()
    st.audio(upload.getvalue())
    if st.button('종 동정 시작', type='primary'):
        temp_path = None
        try:
            model = load_model()
            with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as f:
                f.write(upload.getvalue())
                temp_path = Path(f.name)
            with st.spinner('음원을 읽는 중입니다...'):
                import soundfile as sf
                info = sf.info(str(temp_path))
                if info.duration > 300:
                    st.error('5분 이하로 나눈 파일을 올려 주세요.')
                    st.stop()
                y = load_audio(temp_path)
            segments = windows(y)
            progress = st.progress(0, text='음향 분석 중')
            scores = []
            for i, segment in enumerate(segments):
                x = features(segment)
                score = expit(((x - model['mean']) / model['scale']) @ model['coef'].T + model['intercept'])
                scores.append(score)
                progress.progress((i + 1) / len(segments), text=f'{i+1}/{len(segments)} 구간 분석')
            matrix = np.asarray(scores)
            frame = pd.DataFrame({
                '클래스': model['labels'].tolist(),
                '구간 평균 점수 (%)': matrix.mean(axis=0) * 100,
                '순간 최고 점수 (%)': matrix.max(axis=0) * 100,
                '기준 이상 구간 비율 (%)': (matrix >= 0.5).mean(axis=0) * 100,
            }).sort_values('구간 평균 점수 (%)', ascending=False).reset_index(drop=True)
            st.subheader('분석 결과')
            st.info(f"평균 점수 1위: {frame.iloc[0]['클래스']} — 확정 판정이 아닙니다.")
            st.dataframe(frame.style.format({c: '{:.1f}' for c in frame.columns if c != '클래스'}), hide_index=True, use_container_width=True)
            st.caption('기준 이상 구간 비율은 독립 점수 0.5 이상인 구간 비율입니다. 실제 울음 시간 비율은 아니며, 겹치는 분석 구간을 포함합니다.')
            st.download_button('결과 CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'), file_name='species_results.csv', mime='text/csv')
        except Exception as error:
            st.error(f'분석에 실패했습니다: {error}')
            st.caption('파일 형식을 WAV로 변환하거나 더 짧은 음원으로 시도해 주세요.')
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

st.divider()
st.caption('업로드 음원은 분석용 임시 파일로 처리하고 처리 종료 시 삭제합니다. 공유 PC에서는 업로드 전에 개인정보·대화·음원 이용 권한을 확인하세요. 학습 표본에서의 결과가 새로운 현장음의 정확도를 보장하지는 않습니다.')
