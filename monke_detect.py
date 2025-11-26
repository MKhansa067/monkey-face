import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load gambar monyet
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

# Fungsi bantu: hitung jarak Euclidean antar landmark
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Deteksi expression & gesture
def detect_expression(hand_result, face_result):
    expression = "neutral"

    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            # Hitung jarak bibir atas-bawah dan lebar mulut
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            mouth_gap = distance(upper_lip, lower_lip)

            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            mouth_width = distance(left_mouth, right_mouth)
            mouth_ratio = mouth_gap / mouth_width

            # Deteksi ekspresi dasar wajah
            if mouth_gap > 0.03:
                expression = "surprised"
            elif mouth_ratio > 0.02:
                expression = "smile"
            else:
                expression = "neutral"

            # Fungsi bantu: deteksi apakah mata tertutup
            def is_eye_closed(face_landmarks, left=True):
                if left:
                    top, bottom = face_landmarks.landmark[159], face_landmarks.landmark[145]
                else:
                    top, bottom = face_landmarks.landmark[386], face_landmarks.landmark[374]
                return distance(top, bottom) < 0.01

            left_eye_closed = is_eye_closed(face_landmarks, True)
            right_eye_closed = is_eye_closed(face_landmarks, False)
            one_eye_closed = left_eye_closed != right_eye_closed  # hanya satu mata tertutup
            both_eyes_closed = left_eye_closed and right_eye_closed

            # 5. Jika satu mata tertutup (tanpa bergantung pada gesture tangan)
            if one_eye_closed:
                return "one_eyed"

            # 9. Tambahan: netral + kedua mata tertutup = "eye_closed"
            if expression == "neutral" and both_eyes_closed:
                return "eye_closed"

            # Jika ada tangan, deteksi gesture juga
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    index_pip = hand_landmarks.landmark[6]
                    thumb_tip = hand_landmarks.landmark[4]
                    middle_tip = hand_landmarks.landmark[12]
                    ring_tip = hand_landmarks.landmark[16]
                    pinky_tip = hand_landmarks.landmark[20]
                    wrist = hand_landmarks.landmark[0]

                    # Cek panjang tiap jari untuk tahu lurus atau tidak
                    def is_straight(tip, pip):
                        return distance(tip, wrist) > distance(pip, wrist) * 1.3

                    index_straight = is_straight(index_tip, index_pip)
                    middle_straight = is_straight(middle_tip, hand_landmarks.landmark[10])
                    ring_straight = is_straight(ring_tip, hand_landmarks.landmark[14])
                    pinky_straight = is_straight(pinky_tip, hand_landmarks.landmark[18])
                    thumb_straight = thumb_tip.y < hand_landmarks.landmark[2].y  # kira-kira lurus ke atas

                    # Fungsi bantu: cek apakah tangan setengah mengepal
                    def is_half_clenched(hand_landmarks):
                        folded_count = 0
                        fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
                        for tip, pip in fingers:
                            tip_l = hand_landmarks.landmark[tip]
                            pip_l = hand_landmarks.landmark[pip]
                            if tip_l.y > pip_l.y:
                                folded_count += 1
                        return 2 <= folded_count <= 4

                    half_clenched = is_half_clenched(hand_landmarks)
                    all_straight = all([index_straight, middle_straight, ring_straight, pinky_straight, thumb_straight])

                    # Landmark dagu untuk ekspresi "think"
                    chin = face_landmarks.landmark[152]

                    # 1. Semua jari lurus + neutral = "all_straight"
                    if all_straight and expression == "neutral":
                        return "all_straight"

                    # 2. Half clenched + surprised = "surprised"
                    if half_clenched and expression == "surprised":
                        return "surprised"

                    # 3. Telunjuk lurus, jari lain tidak lurus, dan expression smile = "have_an_idea"
                    if index_straight and not any([middle_straight, ring_straight, pinky_straight]) and expression == "smile":
                        return "have_an_idea"

                    # 4. Telunjuk melengkung dekat dagu = "think"
                    if (not index_straight) and abs(index_tip.x - chin.x) < 0.05 and abs(index_tip.y - chin.y) < 0.05:
                        return "think"
                    
                    # 8. Jari tengah lurus + lainnya tidak = "fuck_gorilla"
                    if middle_straight and not any([index_straight, ring_straight, pinky_straight]):
                        return "fuck_gorilla"

                    # 6. Half clenched + neutral + open eyes = "surprise_side_eye"
                    if half_clenched and expression == "neutral" and not both_eyes_closed:
                        return "surprise_side_eye"

                    # 7. Half clenched + surprised + close eye + close mouth = "eye_closed"
                    if half_clenched and both_eyes_closed and mouth_gap < 0.03:
                        return "eye_closed"

            # 10. Jika tidak ada tangan tapi wajah terdeteksi → gunakan ekspresi wajah
            if not hand_result.multi_hand_landmarks:
                return expression

    # 11. Jika wajah dan tangan dua-duanya tidak terdeteksi → "turn_back"
    if not face_result.multi_face_landmarks and not hand_result.multi_hand_landmarks:
        return "turn_back"

    # 12. Jika tidak masuk kondisi di atas, kembalikan ekspresi default
    return expression


# Jalankan kamera
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
face = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

print("Press ESC to Quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face.process(rgb)

    expression = detect_expression(hand_result, face_result)
    
    # Gambar landmark tangan
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2),
            )

    # Tentukan gambar yang sesuai
    if expression == "surprised":
        overlay = img_surprise
        label = "Monke surprised"
    elif expression == "have_an_idea":
        overlay = img_idea
        label = "Monke have an idea"
    elif expression == "think":
        overlay = img_think
        label = "Monke think"
    elif expression == "all_straight":
        overlay = img_allstraight
        label = "All Hail Hitler"
    elif expression == "one_eyed":
        overlay = img_one_eyed
        label = "Monke one eyed"
    elif expression == "surprise_side_eye":
        overlay = img_surprise_side_eye
        label = "Monke side eye"
    elif expression == "eye_closed":
        overlay = img_eye_closed
        label = "Monke eye closed"
    elif expression == "fuck_gorilla":
        overlay = img_fuck_gorilla
        label = "Monke fuck gorilla"
    elif expression == "turn_back":
        overlay = img_turn_back
        label = "Monke turn back"
    else:
        overlay = img_neutral
        label = "Monke neutral"

    # Cek jika gambar gagal dimuat
    if overlay is None:
        cv2.putText(frame, "Image unavailable!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        overlay_resized = cv2.resize(overlay, (400, 400))
        cv2.imshow("Monke Expression", overlay_resized)

    # Tampilkan pipeline di jendela utama
    # cv2.putText(frame, f"Expression: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Pipeline Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
