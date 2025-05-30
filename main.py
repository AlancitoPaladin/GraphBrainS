import os

import cv2
import numpy as np
from PIL import Image

from imgseg import Imgseg

# ===========================================
# CONFIGURACIÓN Y CREACIÓN DE CARPETAS
# ===========================================
training_path = 'images1/1/Training/glioma'
results_path = 'results'
grid_size = 80

# Crear carpeta de resultados si no existe
if not os.path.exists(results_path):
    os.makedirs(results_path)
    print(f"Creada carpeta: {results_path}")

# Obtener lista de archivos de imagen
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = []

# Buscar archivos de imagen en la carpeta
for file in os.listdir(training_path):
    if any(file.lower().endswith(ext) for ext in image_extensions):
        image_files.append(file)

# Ordenar y tomar solo las primeras 5
image_files.sort()
image_files = image_files[:5]

print(f"Procesando las primeras 5 imágenes de {training_path}")
print(f"Imágenes encontradas: {len(image_files)}")
print(f"Resultados se guardarán en: {results_path}")

# ===========================================
# PROCESAMIENTO Y GUARDADO DE RESULTADOS
# ===========================================
results = []
for i, filename in enumerate(image_files, 1):
    img_path = os.path.join(training_path, filename)

    try:
        print(f"\n=== Procesando imagen {i}/5: {filename} ===")

        # Cargar y redimensionar imagen
        img = Image.open(img_path)
        img = img.resize((320, 240))
        img_array = np.array(img)

        # Procesar con Imgseg
        seg = Imgseg(img_path=img_array, grid_size=grid_size)
        result = seg.seg_rate()

        # Validar que result no es None y tiene el formato correcto
        if result is None:
            print(f"Error: seg_rate() retornó None para {filename}")
            continue

        # Desempaquetar con validación
        if len(result) >= 4:
            u, img_contours, c_rate, u_rate = result[:4]
            # Si hay más elementos, tomarlos también
            otsu_result = result[4] if len(result) > 4 else None
            brain_cavity = result[5] if len(result) > 5 else None
        else:
            print(f"Error: resultado incompleto para {filename}")
            continue

        print(f"Archivo: {filename}")
        print(f"Coverage: {c_rate:.2f}%")
        print(f"Uniformity: {u_rate:.2f}%")

        # ===========================================
        # GUARDAR RESULTADOS EN ARCHIVOS
        # ===========================================
        base_name = os.path.splitext(filename)[0]

        # 1. Guardar imagen original redimensionada
        original_path = os.path.join(results_path, f"{base_name}_original.jpg")
        if len(img_array.shape) == 3:
            cv2.imwrite(original_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(original_path, img_array)

        # 2. Guardar máscara de segmentación
        if u is not None:
            segmentation_mask = (u * 255).astype(np.uint8)
            mask_path = os.path.join(results_path, f"{base_name}_segmentation.jpg")
            cv2.imwrite(mask_path, segmentation_mask)

        # 3. Guardar imagen con contornos
        if img_contours is not None:
            contours_path = os.path.join(results_path, f"{base_name}_contours.jpg")
            if len(img_contours.shape) == 3:
                cv2.imwrite(contours_path, cv2.cvtColor(img_contours, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(contours_path, img_contours)

        # 4. Guardar resultado Otsu (si existe)
        if otsu_result is not None:
            otsu_path = os.path.join(results_path, f"{base_name}_otsu.jpg")
            cv2.imwrite(otsu_path, otsu_result)

        # 5. Guardar cavidad craneal (si existe)
        if brain_cavity is not None:
            cavity_path = os.path.join(results_path, f"{base_name}_brain_cavity.jpg")
            cv2.imwrite(cavity_path, brain_cavity)

        # 6. Guardar datos numéricos
        stats_path = os.path.join(results_path, f"{base_name}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Archivo: {filename}\n")
            f.write(f"Dimensiones: {img_array.shape}\n")
            f.write(f"Coverage Rate: {c_rate:.2f}%\n")
            f.write(f"Uniformity Rate: {u_rate:.2f}%\n")
            f.write(f"Grid Size: {grid_size}\n")
            if u is not None:
                f.write(f"Píxeles segmentados: {np.sum(u > 0.5)}\n")
                f.write(f"Píxeles totales: {u.shape[0] * u.shape[1]}\n")
            else:
                f.write(f"Píxeles segmentados: 0\n")
                f.write(f"Píxeles totales: {img_array.shape[0] * img_array.shape[1]}\n")

        print(f"Archivos guardados para {filename}:")
        print(f"  - {original_path}")
        if u is not None:
            print(f"  - {mask_path}")
        if img_contours is not None:
            print(f"  - {contours_path}")
        if otsu_result is not None:
            print(f"  - {otsu_path}")
        if brain_cavity is not None:
            print(f"  - {cavity_path}")
        print(f"  - {stats_path}")

        # Guardar resultados para resumen
        results.append({
            'filename': filename,
            'coverage': c_rate,
            'uniformity': u_rate,
            'result': result
        })

    except Exception as e:
        print(f"Error procesando {filename}: {str(e)}")
        import traceback

        traceback.print_exc()

# ===========================================
# GUARDAR RESUMEN GENERAL
# ===========================================
summary_path = os.path.join(results_path, 'summary.txt')
with open(summary_path, 'w') as f:
    f.write("RESUMEN DE RESULTADOS DE SEGMENTACIÓN\n")
    f.write("=" * 50 + "\n\n")

    if results:
        total_coverage = sum(r['coverage'] for r in results)
        total_uniformity = sum(r['uniformity'] for r in results)

        f.write(f"Imágenes procesadas exitosamente: {len(results)}\n")
        f.write(f"Coverage promedio: {total_coverage / len(results):.2f}%\n")
        f.write(f"Uniformity promedio: {total_uniformity / len(results):.2f}%\n")
        f.write(f"Grid size utilizado: {grid_size}\n\n")

        f.write("Detalle por imagen:\n")
        for r in results:
            f.write(f"  {r['filename']}: Coverage={r['coverage']:.2f}%, Uniformity={r['uniformity']:.2f}%\n")
    else:
        f.write("No se procesaron imágenes exitosamente.\n")

print(f"\n=== PROCESO COMPLETADO ===")
print(f"Resumen guardado en: {summary_path}")
if results:
    print(f"Total de archivos generados: {len(results) * 6 + 1}")
else:
    print("No se procesaron imágenes exitosamente.")
