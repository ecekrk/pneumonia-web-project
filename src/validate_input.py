from PIL import Image
import numpy as np


def validate_xray_image(image_path):
    """
    Yüklenen görselin chest X-ray olup olmadığını kaba ama daha güçlü kurallarla kontrol eder.
    Not: Bu tıbbi/doğrulayıcı bir model değildir; alakasız görselleri elemek için ön filtredir.
    """

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False, "Yüklenen dosya geçerli bir görüntü olarak açılamadı."

    width, height = img.size

    # 1) Çok küçük görselleri reddet
    if width < 256 or height < 256:
        return False, "Görsel çözünürlüğü çok düşük. Lütfen daha uygun bir akciğer röntgen görüntüsü yükleyin."

    rgb = np.array(img).astype(np.float32)

    # 2) Renk kanalları farkı: X-ray genelde gri tonludur
    rg_diff = np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1]))
    gb_diff = np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2]))
    rb_diff = np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 2]))
    mean_channel_diff = (rg_diff + gb_diff + rb_diff) / 3.0

    if mean_channel_diff > 12:
        return False, "Görsel gri ton ağırlıklı görünmüyor. Lütfen akciğer röntgeni (chest X-ray) yükleyin."

    # 3) Gri görüntü analizi
    gray = img.convert("L")
    gray_np = np.array(gray).astype(np.float32)

    mean_intensity = float(np.mean(gray_np))
    std_intensity = float(np.std(gray_np))

    # Aşırı karanlık / aşırı parlak / çok düz görüntüler reddedilsin
    if mean_intensity < 25 or mean_intensity > 235:
        return False, "Görsel parlaklık dağılımı uygun görünmüyor. Bu görüntü bir chest X-ray olmayabilir."

    if std_intensity < 18:
        return False, "Görsel kontrastı çok düşük. Bu görüntü uygun bir akciğer röntgeni olmayabilir."

    # 4) Kenar yoğunluğu: tamamen düz/boş resimleri ele
    gx = np.abs(np.diff(gray_np, axis=1))
    gy = np.abs(np.diff(gray_np, axis=0))
    edge_strength = float((gx.mean() + gy.mean()) / 2.0)

    if edge_strength < 4:
        return False, "Görsel yapısı çok düz görünüyor. Bu görüntü bir akciğer röntgeni olmayabilir."

    # 5) Orta bölge ile kenar bölgeleri arasında kaba yoğunluk farkı kontrolü
    # Chest X-ray'lerde merkez bölge ve çevre arasında genelde belirli bir yapı farkı olur.
    h, w = gray_np.shape
    center = gray_np[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    border_mask = np.ones_like(gray_np, dtype=bool)
    border_mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = False
    border = gray_np[border_mask]

    center_mean = float(np.mean(center))
    border_mean = float(np.mean(border))
    center_border_diff = abs(center_mean - border_mean)

    if center_border_diff < 2:
        return False, "Görsel yoğunluk yapısı chest X-ray görünümüne yeterince benzemiyor."

    # 6) En-boy oranı çok anormal olmasın
    aspect_ratio = width / height
    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
        return False, "Görsel oranı chest X-ray için uygun görünmüyor."

    return True, "Görsel ön kontrolden geçti."