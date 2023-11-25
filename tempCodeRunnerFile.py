import os
import random
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

# Hàm thực hiện biến đổi Fourier ngắn thời gian (Short Time Fourier Transform)
def shortTimeFourierTransform(sig, frame_size, overlap_fac=0.5, window=np.hanning):
    """
    Tính biến đổi Fourier ngắn thời gian (STFT) của một tín hiệu.

    Tham số:
    - sig (numpy.ndarray): Tín hiệu đầu vào.
    - frame_size (int): Kích thước của các khung (cửa sổ) được sử dụng cho STFT. Xác định số mẫu trong mỗi khung.
    - overlap_fac (float, optional): Hệ số chồng lấp, quyết định sự chồng lấp giữa các khung liên tiếp. Mặc định là 0.5.
    - window (function, optional): Hàm cửa sổ được áp dụng cho mỗi khung trước khi tính biến đổi Fourier. Mặc định là hàm Hanning.
    """
    # Tạo cửa sổ (window) dựa trên hàm cửa sổ được chỉ định (mặc định là Hanning)
    win = window(frame_size)
    
    # Tính toán kích thước bước dựa trên kích thước khung và hệ số chồng lấp
    hop_size = int(frame_size - np.floor(overlap_fac * frame_size))

    # Bổ sung số lượng mẫu không đủ cho một nửa kích thước của khung ở đầu tín hiệu
    samples = np.append(np.zeros(int(np.floor(frame_size/2.0))), sig)
    
    # Xác định số cột cho mảng frames
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1
    
    # Bổ sung số lượng mẫu không đủ ở cuối tín hiệu để xử lý trường hợp chiều dài không phải là bội số chính xác của kích thước bước
    samples = np.append(samples, np.zeros(frame_size))

    # Sử dụng as_strided để tạo mảng 2D frames từ tín hiệu
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frame_size), strides=(samples.strides[0]*hop_size, samples.strides[0])).copy()
    # Áp dụng cửa sổ cho từng khung
    frames *= win

    # Trả về kết quả của biến đổi Fourier rời rạc một chiều của các khung
    return np.fft.rfft(frames)

# Hàm tạo biểu đồ phổ logarit
def logscale_spec(spec, sr=44100, factor=20.):
    """
    Áp dụng biểu đồ logarit cho spectrogram.

    Tham số:
    - spec (numpy.ndarray): Spectrogram cần được chuyển đổi.
    - sr (int, optional): Tần số lấy mẫu của tín hiệu. Mặc định là 44100 Hz.
    - factor (float, optional): Faktor to shift the scale. Mặc định là 20.
    """
    # Lấy kích thước của spectrogram
    timebins, freqbins = np.shape(spec)

    # Tính toán scale dựa trên factor và số lượng bins tần số
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # Tạo một biểu đồ mới với số lượng bins tần số giảm đi
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # Tính toán tần số trung bình cho mỗi bin mới
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

# Hàm vẽ biểu đồ phổ
def plot_spectrogram(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    """
    Vẽ spectrogram và lưu nó xuống đĩa nếu được chỉ định.

    Tham số:
    - audiopath (str): Đường dẫn đến file âm thanh WAV.
    - binsize (int, optional): Kích thước của khung (frame) cho STFT. Mặc định là 2^10.
    - plotpath (str, optional): Đường dẫn để lưu biểu đồ. Nếu không có, biểu đồ sẽ được hiển thị ngay lập tức.
    - colormap (str, optional): Bảng màu cho biểu đồ. Mặc định là "jet".
    """
    # Đọc tín hiệu âm thanh từ file WAV
    samplerate, samples = wav.read(audiopath)
    N = len(samples)
    
    print(f"Tần số lấy mẫu (samplerate): {samplerate} Hz")
    print(f"Tổng số mẫu N: {N}")
    # Tính toán biến đổi Fourier ngắn thời gian (STFT) của tín hiệu
    s = shortTimeFourierTransform(samples, binsize)

    # Áp dụng biểu đồ logarit cho spectrogram
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    # Chuyển đổi amplitudes thành đơn vị dB
    ims = 20.*np.log10(np.abs(sshow)/10e-6)

    timebins, freqbins = np.shape(ims)

    # Vẽ biểu đồ spectrogram
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    # Đặt các nhãn trục x và y dựa trên thời gian và tần số
    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    # Lưu biểu đồ xuống đĩa nếu được chỉ định
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    # Đặt lại biểu đồ để chuẩn bị cho lượt vẽ tiếp theo
    plt.clf()

    return ims

# Hàm để lấy danh sách ngẫu nhiên của một số người
def get_random_people(data_path, num_people=4):
    """
    Lấy danh sách ngẫu nhiên của một số người từ thư mục dữ liệu.

    Tham số:
    - data_path (str): Đường dẫn đến thư mục dữ liệu.
    - num_people (int, optional): Số lượng người muốn chọn ngẫu nhiên. Mặc định là 4.
    """
    # Lấy danh sách tất cả người trong thư mục dữ liệu
    people = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    # Chọn ngẫu nhiên số lượng người được chỉ định
    return random.sample(people, num_people)

# Hàm để tạo ra các spectrogram cho mỗi âm nguyên âm
def generate_spectrograms(data_path, people, vowels=['a', 'i', 'u', 'e', 'o']):
    """
    Tạo ra các spectrogram cho mỗi âm nguyên âm của mỗi người và lưu chúng xuống đĩa.

    Tham số:
    - data_path (str): Đường dẫn đến thư mục chứa dữ liệu.
    - people (list): Danh sách tên người.
    - vowels (list, optional): Danh sách âm nguyên âm cần tạo spectrogram. Mặc định là ['a', 'i', 'u', 'e', 'o'].
    """
    # Duyệt qua từng người và từng nguyên âm để tạo spectrogram
    for person in people:
        person_path = os.path.join(data_path, person)
        for vowel in vowels:
            filename = f"{vowel}.wav"
            filepath = os.path.join(person_path, filename)
            plotpath = f"{person}_{vowel}_spectrogram.png"
            plot_spectrogram(filepath, plotpath=plotpath)

# Đường dẫn đến thư mục chứa dữ liệu
data_path = "C:/Users/minhd/OneDrive/Desktop/xlths bao cao/NguyenAmHuanLuyen-16k"

# Lấy danh sách 4 người ngẫu nhiên
selected_people = get_random_people(data_path, num_people=4)

# Tạo ra các spectrogram cho mỗi âm nguyên âm cho những người được chọn
generate_spectrograms(data_path, selected_people)
