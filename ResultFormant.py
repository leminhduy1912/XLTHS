import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu của từng người và từng nguyên âm
data = {
    "Person 1": {
        "a": [823, 1429, 2972],
        "i": [495, 2313, 3299],
        "u": [503, 812, 2597],
        "e": [707, 1976, 3462],
        "o": [762, 1060, 2821],
    },
    "Person 2": {
        "a": [1003, 1167, 1741],
        "i": [594, 1421, 2758],
        "u": [482, 816, 1421],
        "e": [759, 911, 2225],
        "o": [887, 1179, 2834],
    },
    "Person 3": {
        "a": [790, 1477, 2364],
        "i": [398, 1904, 2727],
        "u": [474, 957, 2183],
        "e": [717, 1723, 2304],
        "o": [762, 1094, 2344],
    },
    "Person 4": {
        "a": [779, 1302, 2743],
        "i": [490, 2108, 2777],
        "u": [473, 792, 2444],
        "e": [744, 1453, 2571],
        "o": [756, 1172, 2727],
    },
}

# Biểu đồ cột cho mỗi người và từng nguyên âm
fig, axes = plt.subplots(nrows=len(data), ncols=len(data["Person 1"]), figsize=(15, 10))

for i, (nguoi, nguyen_am_data) in enumerate(data.items()):
    for j, (nguyen_am, gia_tri) in enumerate(nguyen_am_data.items()):
        axes[i, j].bar(["F1", "F2", "F3"], gia_tri, color=plt.cm.Spectral(i / len(data)))
        axes[i, j].set_title(f"{nguyen_am} - {nguoi}")

# Chỉnh sửa layout
plt.tight_layout()
plt.show()
