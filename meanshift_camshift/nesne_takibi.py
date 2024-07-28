import cv2
import numpy as np

# Kamerayı aç
cap = cv2.VideoCapture(0)

# Bir tane frame oku
ret, frame = cap.read()

if not ret:
    print("Uyarı: Kamera açılmadı.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Yüz tespiti
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects = face_cascade.detectMultiScale(frame)

# Yüz bulunamadıysa uyarı ver
if len(face_rects) == 0:
    print("Uyarı: Yüz bulunamadı.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# İlk yüzün koordinatlarını al
(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)  # Meanshift algoritması girdisi

# Region of interest (ROI)
roi = frame[face_y:face_y + h, face_x:face_x + w]  # ROI = face

# ROI'yi HSV formatına çevir ve histogramı hesapla
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])  # Takip için histogram
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Takip için gerekli durdurma kriterleri
# count = hesaplanacak max öge sayısı
# eps = değişiklik
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Histogramı bir görüntüde bulmak için kullanıyoruz
        # Dinsel karşılaştırma
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.imshow("Takip", img2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

"""""
Kamera Açılışı ve Frame Okuma:
cap = cv2.VideoCapture(0) ile kamerayı açıyoruz.
İlk frame'i okuyarak yüz tespiti yapıyoruz.
Eğer kamera açılmazsa veya yüz bulunamazsa, program sonlanıyor.
Yüz Tespiti:

face_cascade.detectMultiScale(frame) ile yüz tespiti yapıyoruz.
Eğer yüz bulunamazsa, uyarı veriyoruz ve programı sonlandırıyoruz.
ROI ve Histogram:

Tespit edilen yüz bölgesini (ROI) seçip HSV formatına çeviriyoruz.
Bu bölgenin histogramını hesaplayıp normalize ediyoruz.
Takip Döngüsü:

Her frame'de ROI histogramını kullanarak geri projeksiyon hesaplıyoruz.
meanShift algoritması ile takip penceresini güncelliyoruz.
Güncellenen pencereyi dikdörtgen olarak çizip gösteriyoruz.

"""""