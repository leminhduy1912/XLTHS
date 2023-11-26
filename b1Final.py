import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_wideband_spectrogram(file_path, save_path):
    # Đọc file âm thanh WAV
    #y, sr = librosa.load(file_path)
    y, sr = librosa.load(file_path, sr=16000)
    

    print("Dữ liệu âm thanh (y):", y)
    print("Tần số lấy mẫu (sr):", sr)
    # Tính toán phổ băng rộng
    

    #D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    D = librosa.amplitude_to_db(librosa.stft(y, window='hann',hop_length=4000), ref=np.max)

    # Vẽ đồ thị
    plt.figure(figsize=(7, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Wideband Spectrogram')

    # Lưu ảnh phổ xuống đĩa
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # Đóng biểu đồ sau khi lưu

def process_folders(data_path, num_folders=4):
    # Lấy danh sách tất cả các thư mục trong thư mục dữ liệu
    all_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

    # Chọn ngẫu nhiên num_folders thư mục
    selected_folders = random.sample(all_folders, num_folders)

    for folder in selected_folders:
        folder_path = os.path.join(data_path, folder)
        
        # Duyệt qua mỗi file nguyên âm trong thư mục
        for vowel_file in os.listdir(folder_path):
            if vowel_file.lower().endswith(".wav"):
                vowel_path = os.path.join(folder_path, vowel_file)

                # Đặt tên file lưu ảnh phổ
                save_name = f"{folder}_{os.path.splitext(vowel_file)[0]}.png"
                save_path = os.path.join(data_path, "Spectrograms", save_name)

                # Gọi hàm plot_wideband_spectrogram cho mỗi file nguyên âm
                plot_wideband_spectrogram(vowel_path, save_path)

# Thay đổi đường dẫn này thành đường dẫn thực tế của thư mục chứa dữ liệu của bạn
data_folder = "C:/Users/minhd/OneDrive/Desktop/xlths bao cao/NguyenAmHuanLuyen-16k"

# Tạo thư mục Spectrograms nếu nó chưa tồn tại
save_folder = os.path.join(data_folder, "Spectrograms")
os.makedirs(save_folder, exist_ok=True)

# Gọi hàm process_folders để xử lý ảnh phổ cho các thư mục và file nguyên âm ngẫu nhiên
process_folders(data_folder)