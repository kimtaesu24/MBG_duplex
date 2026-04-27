import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os  # 파일 경로 처리를 위해 추가

def visualize_stereo_waveform(file_path):
    # 1. 오디오 파일 로드
    y, sr = librosa.load(file_path, mono=False, sr=None)

    if y.ndim < 2:
        print(f"'{file_path}'는 모노 파일입니다. 스테레오 파일이 필요합니다.")
        return

    # 2. 시각화 설정
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 4))

    librosa.display.waveshow(y[0], sr=sr, ax=ax[0], color='blue')
    ax[0].set(title='Speaker', ylabel='Amplitude')
    ax[0].label_outer()

    librosa.display.waveshow(y[1], sr=sr, ax=ax[1], color='orange')
    ax[1].set(title='Listener', xlabel='Time (s)', ylabel='Amplitude')

    plt.tight_layout()

    # 3. 파일명 추출 및 저장 경로 설정
    # 파일 경로에서 디렉토리, 파일명, 확장자를 분리합니다.
    file_dir = os.path.dirname(file_path)         # 예: 'test_outputs'
    file_base = os.path.basename(file_path)       # 예: 'TS3011c_27_stereo_merged.wav'
    file_name = os.path.splitext(file_base)[0]    # 예: 'TS3011c_27_stereo_merged'

    # 저장될 이미지 파일명 생성 (png 포맷)
    save_path = os.path.join(file_dir, f"{file_name}.png")

    # 그래프 저장 및 출력
    plt.savefig(save_path)
    print(f"이미지가 저장되었습니다: {save_path}")
    plt.show()
    plt.close() # 메모리 관리를 위해 창 닫기

if __name__=="__main__":
    visualize_stereo_waveform('test_outputs/TS3011c_33_stereo_merged.wav')