import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu của từng người và từng nguyên âm
data = {
    "Nguoi 1": {
        "Âm a": [823, 1429, 2972],
        "Âm i": [495, 2313, 3299],
        "Âm u": [503, 812, 2597],
        "Âm e": [707, 1976, 3462],
        "Âm o": [762, 1060, 2821],
    },
    "Nguoi 2": {
        "Âm a": [1003, 1167, 1741],
        "Âm i": [594, 2758, 1421],
        "Âm u": [482, 816, 1421],
        "Âm e": [759, 911, 2225],
        "Âm o": [887, 1179, 2834],
    },
    "Nguoi 3": {
        "Âm a": [790, 1477, 2364],
        "Âm i": [398, 1904, 2727],
        "Âm u": [474, 957, 2183],
        "Âm e": [717, 1723, 2304],
        "Âm o": [762, 1094, 2344],
    },
    "Nguoi 4": {
        "Âm a": [779, 1302, 2743],
        "Âm i": [490, 2108, 2777],
        "Âm u": [473, 792, 2444],
        "Âm e": [744, 1453, 2571],
        "Âm o": [756, 1172, 2727],
    },
}

# Biểu đồ cột cho mỗi người và từng nguyên âm
fig, axes = plt.subplots(nrows=len(data), ncols=len(data["Nguoi 1"]), figsize=(15, 10))

for i, (nguoi, nguyen_am_data) in enumerate(data.items()):
    for j, (nguyen_am, gia_tri) in enumerate(nguyen_am_data.items()):
        axes[i, j].bar(["F1", "F2", "F3"], gia_tri, color=plt.cm.Spectral(i / len(data)))
        axes[i, j].set_title(f"{nguyen_am} - {nguoi}")

# Chỉnh sửa layout
plt.tight_layout()
plt.show()
