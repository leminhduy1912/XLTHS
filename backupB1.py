import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
def plot_wideband_spectrogram(file_path):
    # Đọc file âm thanh WAV
    y, sr = librosa.load(file_path)

    # Tính toán phổ băng rộng
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    # Vẽ đồ thị
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Wideband Spectrogram')
    plt.show()

# Gọi hàm với đường dẫn của file WAV
file_path = "C:\\Users\\minhd\\OneDrive\\Desktop\\xlths bao cao\\NguyenAmHuanLuyen-16k\\25MLM\\a.wav"  # Thay đổi thành đường dẫn thực tế của file WAV của bạn
plot_wideband_spectrogram(file_path)