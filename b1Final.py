import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import random
def plot_wideband_spectrogram(folder_name, vowel_file, file_path, save_path):
    y, sr = librosa.load(file_path, sr=16000)

    print("Dữ liệu âm thanh (y):", y)
    print("Tần số lấy mẫu (sr):", sr)
    num_samples = len(y)
    print(f"Số mẫu trong tín hiệu: {num_samples}")

    D = librosa.amplitude_to_db(librosa.stft(y, window='hann'), ref=np.max)

    # Lấy tên folder và tên file nguyên âm
    folder_name = os.path.basename(folder_name)
    vowel_file_name = os.path.splitext(vowel_file)[0]

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    # Bổ sung tên folder và tên file vào đồ thị
    plt.title(f'Wideband Spectrogram - {folder_name}_{vowel_file_name}')

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def process_folders(data_path, selected_folders):
    for folder in selected_folders:
        folder_path = os.path.join(data_path, folder)
        
        for vowel_file in os.listdir(folder_path):
            if vowel_file.lower().endswith(".wav"):
                vowel_path = os.path.join(folder_path, vowel_file)

                save_name = f"{folder}_{os.path.splitext(vowel_file)[0]}.png"
                save_path = os.path.join(data_path, "Spectrograms", save_name)

                plot_wideband_spectrogram(folder, vowel_file, vowel_path, save_path)

# Đường dẫn của thư mục chứa dữ liệu
data_folder = "C:/Users/minhd/OneDrive/Desktop/xlths bao cao/NguyenAmHuanLuyen-16k"

# Danh sách 4 thư mục bạn muốn xử lý
selected_folders = ["25MLM", "23MTL", "24FTL", "30FTN"]

# Tạo thư mục Spectrograms nếu nó chưa tồn tại
save_folder = os.path.join(data_folder, "Spectrograms")
os.makedirs(save_folder, exist_ok=True)

# Gọi hàm process_folders để xử lý ảnh phổ cho các thư mục và file nguyên âm được chỉ định
process_folders(data_folder, selected_folders)
