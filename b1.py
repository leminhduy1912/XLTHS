import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import python_speech_features
import pandas as pd
import seaborn as sns

# Hàm lấy đường dẫn đến thư mục của một người nói
def get_person_path(person):
    return "C:/Users/minhd/OneDrive/Desktop/xlths bao cao/NguyenAmHuanLuyen-16k/" + person

# Hàm lấy đường dẫn đến file âm thanh của một nguyên âm
def get_vowel_path(person, vowel):
    return f"{get_person_path(person)}/{vowel}.wav"

# Hàm hiển thị ảnh phổ từ file âm thanh
def show_spectrogram(person, vowel, audio, sample_rate):
    # Lọc tiếng ồn
    audio = scipy.signal.filtfilt([1, 0.95], [1, -0.95], audio)

    # Chuyển đổi ma trận sos để khắc phục lỗi
    sos = scipy.signal.butter(10, [300, 2800], btype='band', fs=sample_rate, output='sos')
    sos = sos[:, :6]

    # Cách ly nguồn âm
    audio = scipy.signal.sosfilt(sos, audio)

    # Tạo phổ tiếng nói
    plt.figure(figsize=(8, 6))
    plt.specgram(audio, Fs=sample_rate)
    plt.title(f'Spectrogram - Person: {person}, Vowel: {vowel}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Hiển thị ảnh
    plt.show()

# Tạo DataFrame để lưu trữ dữ liệu
data = {'Person': [], 'Vowel': [], 'F1': [], 'F2': [], 'F3': []}

# Chọn ngẫu nhiên 4 người nói từ 21 người nói
people = ["1", "5", "6", "8"]

# Xử lý dữ liệu
for person in people:
    # Lặp qua các nguyên âm
    for vowel in ["a", "i", "u", "e", "o"]:
        # Đọc file âm thanh WAV
        sample_rate, audio = scipy.io.wavfile.read(get_vowel_path(person, vowel))

        # Hiển thị ảnh phổ và xử lý dữ liệu
        show_spectrogram(person, vowel, audio, sample_rate)

        # Đo tần số formant
        formant = python_speech_features.mfcc(audio, samplerate=sample_rate, nfft=1024)

        # Thêm dữ liệu vào DataFrame
        data['Person'].extend([person] * len(formant[0]))
        data['Vowel'].extend([vowel] * len(formant[0]))
        data['F1'].extend(formant[0])
        data['F2'].extend(formant[1])
        data['F3'].extend(formant[2])

# Tạo DataFrame từ dữ liệu
df = pd.DataFrame(data)

# Vẽ biểu đồ sử dụng Seaborn
plt.figure(figsize=(12, 8))
sns.boxplot(x='Vowel', y='F1', data=df, hue='Person', palette='Set3')
plt.title('Boxplot of F1 for each Vowel')
plt.show()
