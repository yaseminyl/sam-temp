from ultralytics import SAM
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCV, resimleri işlemek için

# Modeli yükle
model = SAM("sam2.1_b.pt")

# Model bilgilerini görüntüle (isteğe bağlı)
model.info()

# İnferans çalıştır (bbox prompt ile)
results = model("frame_0001.png")

print(results)

# Orijinal resmi yükleyin
orig_img = cv2.imread("frame_0001.png")  # OpenCV ile orijinal resmi yükleyin

# Maskeleri numpy dizisine dönüştür
if results[0].masks:  # Maskeler varsa
    
    # Orijinal resmi kopyalayın, üzerine maskeleri ekleyeceğiz
    mask_overlay = orig_img.copy()

    for idx, mask in enumerate(results[0].masks):  # Listeyi döngüyle kontrol et
        mask_data = mask.data.cpu().numpy()  # Her bir maskeyi al
        print(f"Maske {idx+1} şekli: {mask_data.shape}")

        # Maskeyi 2D forma getir
        mask_data = mask_data[0]  # İlk boyutu çıkar (1, 1080, 1920 -> 1080, 1920)

        # Maskeyi orijinal görüntü üzerine uygulayın (şeffaflık ile)
        mask_colored = np.zeros_like(orig_img)  # Maskenin boyutlarında boş bir resim oluştur
        mask_colored[mask_data == 1] = [0, 255, 0]  # Maskedeki 1'leri yeşil renkle doldurun

        # Maskeyi orijinal görüntüye şeffaf bir şekilde ekleyin
        mask_overlay = cv2.addWeighted(mask_overlay, 1, mask_colored, 0.5, 0)

    # Maskelerin bulunduğu resimi göster
    plt.imshow(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))  # BGR'yi RGB'ye dönüştür
    plt.title("Tüm Maskeler")
    plt.axis('off')  # Eksenleri kaldır
    plt.show()

    # Maskeleri kaydedin
    plt.imsave("ALL.png", cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))  # Tüm maskeleri orijinal resim üzerinde kaydet
else:
    print("Hiç maske bulunamadı.")
