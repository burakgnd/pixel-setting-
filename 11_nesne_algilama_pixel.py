from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

model = YOLO('yolov8s.pt')

generator = sv.get_video_frames_generator('/Users/soylu/Desktop/Proje_Soylu/Alan Tespiti Ve Bildirim /veri_seti/a6.mp4')
iterator = iter(generator)
frame = next(iterator)

results = model(frame, imgsz=1280)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoundingBoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
frame = box_annotator.annotate(scene=frame, detections=detections)
frame = label_annotator.annotate(scene=frame, detections=detections)

plt.imshow(frame)
plt.show()
