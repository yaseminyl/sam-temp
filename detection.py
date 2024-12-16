import sys
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# SAM2 modül yolunu ekleme
sys.path.append('/home/yasemingumus/sam2')

# predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")

video_path = "right1.mp4"

input_prompts = {
    "points": [[920, 470], [1000, 500]],
    "labels": [1, 1]  # Her nokta için etiket (1: foreground, 0: background)
}

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device='cpu')

image_path = "frame_0001.png"

image = cv2.imread(image_path)  # Görüntüyü yükleyin
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR'yi RGB'ye dönüştürün

input_point = np.array([[1000, 500]])
input_label = np.array([1])  # Nokta etiketi (1: foreground, 0: background)

def show_masks(image, masks, scores, point_coords=None, input_labels=None, borders=False):
    """
    Maskeleri OpenCV ile görüntüler.
    
    Parameters:
    - image: Orijinal görüntü (RGB formatında)
    - masks: Tahmin edilen maskeler (num_masks x H x W)
    - scores: Maskelerin skoru
    - point_coords: Kullanıcı tarafından belirtilen noktalar (isteğe bağlı)
    - input_labels: Kullanıcı tarafından belirtilen etiketler (isteğe bağlı)
    - borders: Maskelerin sınırlarını çizme (isteğe bağlı)
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV RGB'yi BGR'ye dönüştürür
    
    for i, mask in enumerate(masks):
        color_mask = np.zeros_like(image_bgr)
        
        # Maskeyi renklendir
        color_mask[mask == 1] = [0, 255, 0]  # Yeşil renk
        image_bgr = cv2.addWeighted(image_bgr, 1, color_mask, 0.5, 0)  # Maskeyi transparan olarak ekle
        
        # Maskenin skorunu yaz
        text = f"Score: {scores[i]:.2f}"
        cv2.putText(image_bgr, text, (point_coords[0][0], point_coords[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(image_bgr, [contour], -1, (0, 0, 255), 2)  # Kırmızı sınırlar

    # Noktaları ve etiketleri göster
    if point_coords is not None:
        for i, coord in enumerate(point_coords):
            cv2.circle(image_bgr, tuple(coord), 5, (0, 0, 255), -1)  # Kırmızı noktalar
            cv2.putText(image_bgr, str(input_labels[i]), tuple(coord), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite("savedImage.jpg", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Görüntü üzerinde segmentasyon yapmak
predictor.set_image(image)

# Maskeleri tahmin etme
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    # multimask_output=True,
)

# Maskeleri sıralama (en yüksek skorla en iyi maskeyi önce alacak şekilde)
# sorted_ind = np.argsort(scores)[::-1]
# masks = masks[sorted_ind]
# scores = scores[sorted_ind]
# logits = logits[sorted_ind]

print(masks.shape)

# Maskeleri gösterme
plt.figure(figsize=(10, 10))
plt.imshow(image)
# Maskeleri ve noktaları görüntüleme
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
plt.axis('on')
plt.show()

# Maskeleri ve skorları yazdırma
print("Masks shape:", masks.shape)
print("Scores shape:", scores.shape)
print("Logits shape:", logits.shape)






# Video işleme
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Video dosyası açılamadı!")
        return

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=False)

        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, input_prompts)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            ret, frame = cap.read()
            if not ret:
                break

            if masks is not None:
                for mask in masks:
                    mask = mask.cpu().numpy().astype("uint8") * 255
                    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

            cv2.imshow("SAM2 Segmentasyon", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Segmentasyon işlemini başlat
# process_video(video_path)
