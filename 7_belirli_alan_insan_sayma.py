import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')

polygon = np.array([
    [400, 400],
    [400, 100],
    [550, 100],
    [550, 400]
])

video_path = 'soylu.mp4'
video_info = sv.VideoInfo.from_video_path(video_path)

zone = sv.PolygonZone(polygon=polygon)

box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.4)  # Etiket boyutunu küçülttüm
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=2, text_thickness=2, text_scale=0.8)  # Etiket boyutunu küçülttüm

output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.width, video_info.height))

generator = sv.get_video_frames_generator(video_path)

for frame in generator:
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)

    detections = detections[detections.class_id == 0] # insan nesnesinin nesne etiketi 0 Coco kütüphanesinden bakılıp bulunabilir 

    zone.trigger(detections=detections)

    labels = [f"{model.names[class_id]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)

    out.write(frame)

    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
