# monkey-face
## 1. Cara download program ini  
- Klik tombol **Code** yang berwarna hijau.
- Klik **Download ZIP**, setelah itu ekstrak pada folder lokal anda.  
<div align="center"><img width="553" height="436" alt="image" src="https://github.com/user-attachments/assets/33992be0-cc97-476b-a6a3-e90e6d60df9c" /></div>

## 2. Tools yang dibutuhkan  
### Perangkat Lunak  
- VSCode (text editor)
- Python versi 3.13.0 (recommended)
### Library Python (install via terminal VSCode)
- pip install opencv-python
- pip install mediapipe
- pip install numpy
<div align="center"><img width="562" height="21" alt="image" src="https://github.com/user-attachments/assets/a077e85a-4f4c-4d45-b0f1-ebc4e4d2c223" /></div>

## 3. Pembuatan program per bagian
### Fungsi Library  
- cv2(opencv): Untuk memproses video dari kamera, membaca gambar monyet, menggambar landmark tangan, dan menampilkan jendela output visual.
- mediapipe: memproses data video untuk mendeteksi objek, gerakan tubuh, wajah, tangan, dan titik-titik kunci lainnya.
- numpy: operasi matematika (penjumlahan, pengurangan, perkalian), analisis data (statistik, aljabar linear), dan manipulasi array seperti mengubah bentuk (reshape) atau menggabungkannya (concatenate).
```
import cv2
import mediapipe as mp
import numpy as np
```
