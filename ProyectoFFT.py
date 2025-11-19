"""
------ Iker Garcia  ------
--------- Auf Das ---------
----------- FFT -----------
-------- 08/11/2025 --------
"""
# ------- Main Library -------
# pip install numpy imageio matplotlib reportlab customtkinter pillow
import os
import math
import shutil
import tempfile
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import gc

from reportlab.platypus import Table
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet



# ---------- Class ----------
# -------- Variables --------
MAX_PIXELS = 3_000_000     # máximo número de píxeles permitidos en la FFT final (seguro)
MAX_DIM = 1024             # si la mayor dimensión > MAX_DIM, downsampreamos la imagen antes de procesar
TMP_DIR = r"C:\Users\dasre\Documents\Knowledge-db\ghFiles\pycodes\tmp_fft_safe"   # carpeta temporal para imágenes usadas en el PDF


# --------- Function ---------
def safe_downsample_image_if_needed(img):
    """Si la imagen es muy grande en píxeles, la reduce manteniendo aspect ratio."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_DIM:
        return img, 1.0

    scale0 = MAX_DIM / float(max_dim)
    new_h = int(round(h * scale0))
    new_w = int(round(w * scale0))

    # Simple resize usando imageio (PIL backend) para mantenerlo sencillo
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img.astype(np.uint8))
    pil = pil.resize((new_w, new_h), resample=PILImage.BILINEAR)
    img_small = np.array(pil).astype(np.float32)
    return img_small, scale0


def compute_safe_scale(h, w, requested_scale):
    """Devuelve (effective_scale, was_capped_bool) asegurando que new_h*new_w <= MAX_PIXELS."""
    if requested_scale <= 0:
        return requested_scale, False

    new_h = int(round(h * requested_scale))
    new_w = int(round(w * requested_scale))
    future_pixels = new_h * new_w

    if future_pixels <= MAX_PIXELS:
        return requested_scale, False

    # calcular scale máximo permitido
    max_scale = math.sqrt(MAX_PIXELS / float(h * w))
    max_scale = max_scale if max_scale > 0 else 1e-6

    # asegurar que new dimension mínima sea 2
    if int(round(h * max_scale)) < 2 or int(round(w * max_scale)) < 2:
        # demasiado pequeño, devolvemos scale que preserve al menos 2 píxeles
        safe_scale_h = 2.0 / float(h)
        safe_scale_w = 2.0 / float(w)
        max_scale = max(safe_scale_h, safe_scale_w)

    return max_scale, True

def fft_resize_and_save(img, scale, tmp_dir, basename_label):
    note = ""
    img_proc, input_downscale = safe_downsample_image_if_needed(img)
    if input_downscale != 1.0:
        note += f"Input downsampled x{input_downscale:.3f} para seguridad. "

    h, w = img_proc.shape[:2]

    eff_scale, was_capped = compute_safe_scale(h, w, scale)
    if was_capped:
        note += f"Scale pedido {scale} fue capado a {eff_scale:.4f} para evitar OOM. "

    new_h = max(2, int(round(h * eff_scale)))
    new_w = max(2, int(round(w * eff_scale)))

    resized_path = os.path.join(tmp_dir, f"{basename_label}_resized_.png")

    if img_proc.ndim == 2:
        img_proc = img_proc[:, :, None]

    ch = img_proc.shape[2]
    channel0 = img_proc[:, :, 0]

    F = np.fft.fftshift(np.fft.fft2(channel0))

    if eff_scale >= 1.0:
        F_new = np.zeros((new_h, new_w), dtype=complex)
        h1 = new_h // 2 - h // 2
        w1 = new_w // 2 - w // 2
        F_new[h1:h1 + h, w1:w1 + w] = F
    else:
        crop_h = new_h // 2
        crop_w = new_w // 2
        center_h = h // 2
        center_w = w // 2
        r0 = max(0, center_h - crop_h)
        r1 = min(h, center_h + crop_h)
        c0 = max(0, center_w - crop_w)
        c1 = min(w, center_w + crop_w)
        F_new = F[r0:r1, c0:c1]

    resized_channels = []
    for c in range(ch):
        Fc = np.fft.fftshift(np.fft.fft2(img_proc[:, :, c]))

        if eff_scale >= 1.0:
            F_big = np.zeros((new_h, new_w), dtype=complex)
            h1 = new_h // 2 - h // 2
            w1 = new_w // 2 - w // 2
            F_big[h1:h1 + h, w1:w1 + w] = Fc
            rec = np.fft.ifft2(np.fft.ifftshift(F_big))
        else:
            crop_h = new_h // 2
            crop_w = new_w // 2
            center_h = h // 2
            center_w = w // 2
            r0 = max(0, center_h - crop_h)
            r1 = min(h, center_h + crop_h)
            c0 = max(0, center_w - crop_w)
            c1 = min(w, center_w + crop_w)
            Fc_crop = Fc[r0:r1, c0:c1]
            rec = np.fft.ifft2(np.fft.ifftshift(Fc_crop))

        rec = np.abs(rec)
        if rec.max() > 0:
            rec = rec * (255.0 / rec.max())
        resized_channels.append(rec.astype(np.uint8))

    resized_img = np.stack(resized_channels, axis=2).astype(np.uint8)
    imageio.imwrite(resized_path, resized_img)

    del F, F_new, resized_channels, resized_img, img_proc
    gc.collect()

    return resized_path, note, eff_scale


def generate_fft_report_safe(input_folder, output_pdf, scales):
    """
    Versión segura que procesa imagen por imagen y escala por escala,
    generando un PDF con los resultados sin consumir demasiada RAM.
    """
    # preparar tmp dir
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    story = []

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_ext)]
    if not files:
        print("No hay imágenes en el folder:", input_folder)
        return


    story.append(Paragraph(f"<b>Reporte de Image Upscaling</b>", styles['Heading2']))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"<b>Equipo: Das, Sara, Iker Garcia, Angel:</b>", styles['Heading2']))
    story.append(Spacer(1, 6))

    for filename in files:
        path = os.path.join(input_folder, filename)
        print("Procesando:", filename)
        img = imageio.imread(path).astype(np.float32)

        story.append(Paragraph(f"<b>Imagen:</b> {filename}", styles['Heading2']))
        story.append(Spacer(1, 6))

        # mostrar la miniatura original
        thumb_path = os.path.join(TMP_DIR, f"{filename}_orig.png")
        # crear thumbnail seguro
        thumb_img, _ = safe_downsample_image_if_needed(img)
        imageio.imwrite(thumb_path, thumb_img.astype(np.uint8))
        story.append(Image(thumb_path, width=250, height=250))
        story.append(Spacer(1, 8))

        for scale in scales:
            basename_label = f"{os.path.splitext(filename)[0]}_scale{scale}"
            resized_p, note, eff_scale = fft_resize_and_save(
                img, scale, TMP_DIR, basename_label
            )


            story.append(Paragraph(f"<b>Scale pedido:</b> {scale} <b>· escala efectiva:</b> {eff_scale:.4f}", styles['Normal']))
            if note:
                story.append(Paragraph(f"<i>{note}</i>", styles['Italic']))
            story.append(Spacer(1, 6))

            # ========== BUILD FFT SPECTRA AND COMPARISON ==========

            # Load resized (upscaled) image
            resized_img = imageio.imread(resized_p).astype(np.float32)

            # -------- Convert to grayscale for FFT --------
            def to_gray(x):
                if len(x.shape) == 3:
                    return np.mean(x, axis=2)
                return x

            orig_gray = to_gray(img)
            up_gray   = to_gray(resized_img)

            # -------- FFT original ----------
            F_orig = np.fft.fftshift(np.fft.fft2(orig_gray))
            mag_orig = 20 * np.log(np.abs(F_orig) + 1)

            spec_a = os.path.join(TMP_DIR, f"tmp {basename_label}.png")
            plt.figure()
            plt.imshow(mag_orig, cmap='gray')
            plt.title("Original FFT Spectrum")
            plt.axis('off')
            plt.savefig(spec_a, bbox_inches='tight')
            plt.close()

            # -------- FFT upscaled ----------
            F_up = np.fft.fftshift(np.fft.fft2(up_gray))
            mag_up = 20 * np.log(np.abs(F_up) + 1)

            spec_b = os.path.join(TMP_DIR, f"tmp {basename_label}.png")
            plt.figure()
            plt.imshow(mag_up, cmap='gray')
            plt.title("Upscaled FFT Spectrum")
            plt.axis('off')
            plt.savefig(spec_b, bbox_inches='tight')
            plt.close()



            # Insertar espectros y comparación al PDF
            try:
                # Insert into PDF
                spectra_table = Table([
                    [
                        Image(spec_b, width=230, height=230),
                        Image(spec_a, width=230, height=230)
                    ]
                ])
                story.append(spectra_table)
                story.append(Spacer(1, 8))

                story.append(Spacer(1, 12))
 
            except Exception as e:
                story.append(
                    Paragraph(f"<i>Error insertando imágenes al PDF: {e}</i>", styles['Normal'])
                )

          

            # colecta basura
            gc.collect()

        story.append(Spacer(1, 20))

    # construir PDF
    doc.build(story)
    print("\n✅ PDF generado en:", output_pdf)

