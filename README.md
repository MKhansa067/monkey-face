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
```terminal
pip install opencv-python mediapipe numpy
```
<div align="center"><img width="562" height="21" alt="image" src="https://github.com/user-attachments/assets/a077e85a-4f4c-4d45-b0f1-ebc4e4d2c223" /></div>

## 3. Pembuatan program per bagian
### Fungsi Library  
- cv2(opencv): Untuk memproses video dari kamera, membaca gambar monyet, menggambar landmark tangan, dan menampilkan jendela output visual.
- mediapipe: Memproses data video untuk mendeteksi objek, gerakan tubuh, wajah, tangan, dan titik-titik kunci lainnya.
- numpy: operasi matematika (penjumlahan, pengurangan, perkalian), analisis data (statistik, aljabar linear), dan manipulasi array seperti mengubah bentuk (reshape) atau menggabungkannya (concatenate).

Pada kode seperti ini:
```python
import cv2
import mediapipe as mp
import numpy as np
```

### Inisialisasi Mediapipe
Program ini menggunakan:
- mp.solutions.hands → mendeteksi posisi jari tangan
- mp.solutions.face_mesh → mendeteksi ekspresi wajah (mulut, mata, dagu)
- mp.solutions.drawing_utils → menggambar titik & garis landmark
- mp.solutions.drawing_styles → style untuk rendering landmark
```python
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
```

### Load Gambar Monke
Semua gambar disiapkan untuk ditampilkan sesuai ekspresi:
- Menghadap kebelakang
- Jari tengah
- Kedua mata tertutup
- dst.    
```python
img_turn_back = cv2.imread("images/monke9.jpeg")  # monke to the back
img_fuck_gorilla = cv2.imread("images/monke8.jpeg") # gorilla fuck emote
img_eye_closed = cv2.imread("images/monke7.jpeg")    # kedua mata tertutup
img_surprise_side_eye = cv2.imread("images/monke6.jpeg")  # mata samping terbuka + surprised
img_one_eyed = cv2.imread("images/monke5.jpeg")  # satu mata tertutup
img_surprise = cv2.imread("images/monke4.jpeg")   # surprised
img_idea = cv2.imread("images/monke3.jpeg")       # punya ide
img_think = cv2.imread("images/monke2.jpeg")      # think
img_neutral = cv2.imread("images/monke1.jpeg")    # neutral
img_allstraight = cv2.imread("images/hitler5.jpg")  # semua jari lurus + neutral
```

### Fungsi menghitung jarak landmark
Fungsi ini penting untuk:
- Menentukan apakah mulut terbuka / tersenyum
- Mengecek mata tertutup
- Mengevaluasi apakah jari “lurus” atau “melengkung”  

Pada kode seperti dan menyerupai ini:
```python
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
```

### Fungsi utama deteksi
Bagian ini adalah pusat logika program.  
**Deteksi Wajah:**  
- Mouth Gap → cek mulut terbuka → ekspresi “surprised”
- Mouth Ratio → cek tersenyum → ekspresi “smile”
- Eye Distance → mendeteksi mata tertutup
- One-eyed / eye-closed
- Dst.  
**Deteksi Gesture Tangan:**  
- Fungsi is_straight() → cek jari lurus
- Fungsi is_half_clenched() → cek tangan menggenggam setengah
- Dst.  
**Alur final:**  
- Jika wajah terdeteksi → analisis ekspresi
- Jika tangan terdeteksi → analisis gesture
- Jika keduanya hilang → tampilkan "turn back"

Pada kode seperti dan menyerupai ini:  
```python
def detect_expression(hand_result, face_result):
```

### Membuka Kamera
Mengaktifkan webcam default.
```
cap = cv2.VideoCapture(0)
```
### Proses Frame by Frame

### Menentukan Gambar sesuai Ekspresi

### Menampilkan Output

### Keluar Program

## Cara menjalankan program
- Buka folder di VSCode
- Buka terminal
- Jalankan
```terminal
python 
