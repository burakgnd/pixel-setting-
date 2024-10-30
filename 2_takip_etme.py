import cv2
import numpy as np
from ultralytics import YOLO

# YOLO modeli yükleniyor
model = YOLO('yolov8n.pt')

# Video açılıyor
cap = cv2.VideoCapture('/Users/soylu/Desktop/Proje_Soylu/Alan Tespiti Ve Bildirim /veri_seti/a6.mp4')

tracker = None  # Takipçi başlangıçta None olarak ayarlanıyor
tracking_started = False
polygon_area_active = False  # Poligon içinde mi kontrolü için flag
in_polygon = False  # Nesnenin poligon içinde olup olmadığını takip etmek için

# Takip edilecek alanı tanımlayan polygon
polygon = np.array([
    [200, 400],
    [200, 100],
    [400, 100],
    [400, 400]
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunamadı!")
        break

    # Polygonu çerçeveye çiz (görsel amaçlı)
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)

    if not tracking_started:  # Eğer takip başlamamışsa, YOLO ile nesne tespiti yap
        results = model.predict(source=frame)
        for result in results[0].boxes:
            box = result.xyxy[0]  # Tahmin edilen sınırlayıcı kutunun koordinatları
            cls = result.cls  # Sınıf etiketi

            # Sadece insan sınıfı için kontrol (genellikle 0)
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Nesnenin merkezini hesapla

                # Nesnenin merkezi polygon içinde mi kontrol et
                if cv2.pointPolygonTest(polygon, bbox_center, False) >= 0:
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracker = cv2.TrackerCSRT_create()  # Takipçiyi her tespit için yeniden başlat
                    tracker.init(frame, bbox)  # Tespit edilen nesneyle izlemeye başla
                    tracking_started = True
                    polygon_area_active = True  # Nesne poligon içinde
                    break
    else:
        success, bbox = tracker.update(frame)  # İzleme başarılı mı kontrol et
        if success:
            # İzleme başarılı ise nesneyi çerçeve içine al
            x, y, w, h = map(int, bbox)
            bbox_center = (x + w // 2, y + h // 2)

            # Nesnenin merkezi poligon içinde mi kontrol et
            if cv2.pointPolygonTest(polygon, bbox_center, False) >= 0:
                polygon_area_active = True  # Nesne hala poligon içinde
                # "1" yaz (nesne poligon içinde)
                cv2.putText(frame, "1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                in_polygon = True
                # Takip etiketi göster
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            else:
                polygon_area_active = False  # Nesne poligondan çıktı
                if in_polygon:
                    # "0" yaz (nesne poligondan çıkınca)
                    cv2.putText(frame, "0", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    in_polygon = False
        else:
            # İzleme başarısız olursa "Tracking Lost" yaz
            cv2.putText(frame, "Tracking Lost, Retrying...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            tracking_started = False  # Takibi yeniden başlatmak için sıfırla
            tracker = None  # Takipçiyi sıfırla, yeni tespit için
            
            # Hedef kaybolduğunda hemen YOLO'yu tekrar çalıştır:
            results = model.predict(source=frame)  # YOLO ile yeniden tespit yapılıyor
            found_human = False  # İnsan bulundu mu kontrolü için
            for result in results[0].boxes:
                box = result.xyxy[0]
                cls = result.cls

                # Sadece insan sınıfını takip et
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Nesnenin merkezi polygon içinde mi kontrol et
                    if cv2.pointPolygonTest(polygon, bbox_center, False) >= 0:
                        found_human = True  # İnsan bulundu
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        tracker = cv2.TrackerCSRT_create()  # Takipçi yeniden başlatılıyor
                        tracker.init(frame, bbox)
                        tracking_started = True
                        break
            
            if not found_human:
                cv2.putText(frame, "No Human Detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
