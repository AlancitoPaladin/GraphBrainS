import cv2
import numpy as np
from scipy import ndimage
from skimage import filters

import morphsnakes as ms


def test_MorphACWE(imgcolor):
    """Función principal mejorada para segmentación robusta - SIN VISUALIZACIÓN"""
    print('Iniciando segmentación MorphACWE...')

    # Validar entrada
    if imgcolor is None or imgcolor.size == 0:
        print("Error: Imagen vacía o None")
        # Retornar valores por defecto en lugar de None
        dummy_shape = (240, 320)
        dummy_u = np.zeros(dummy_shape, dtype=np.float64)
        dummy_contours = np.zeros((*dummy_shape, 3), dtype=np.uint8)
        dummy_otsu = np.zeros(dummy_shape, dtype=np.uint8)
        dummy_cavity = np.zeros(dummy_shape, dtype=np.uint8)
        return dummy_u, dummy_contours, dummy_otsu, dummy_cavity

    # Normalizar imagen
    if imgcolor.max() > 1.0:
        imgcolor = imgcolor / 255.0

    # Convertir a escala de grises si es necesario
    if len(imgcolor.shape) == 3:
        imggray = rgb2gray(imgcolor)
        print(f"Imagen RGB convertida a grises. Shape: {imggray.shape}")
    else:
        imggray = imgcolor.copy()
        print(f"Imagen en escala de grises. Shape: {imggray.shape}")

    print(f"Rango de intensidades: [{imggray.min():.3f}, {imggray.max():.3f}]")
    print(f"Intensidad media: {imggray.mean():.3f}")

    # Para debugging, convertir a uint8 para análisis
    gray_uint8 = (imggray * 255).astype(np.uint8) if imggray.max() <= 1.0 else imggray.astype(np.uint8)

    # Intentar método sofisticado primero
    try:
        otsu_result, brain_cavity = improved_otsu_threshold(gray_uint8)

        if np.sum(otsu_result > 0) == 0:
            print("Método sofisticado falló, usando segmentación simple...")
            otsu_result = simple_brain_segmentation(gray_uint8)
            brain_cavity = otsu_result.copy()  # Usar la misma máscara

    except Exception as e:
        print(f"Error en método sofisticado: {e}")
        print("Usando segmentación simple...")
        otsu_result = simple_brain_segmentation(gray_uint8)
        brain_cavity = otsu_result.copy()

    # Verificar que tenemos una inicialización válida
    if np.sum(otsu_result > 0) == 0:
        print("ADVERTENCIA: Todas las segmentaciones fallaron, creando inicialización mínima")
        h, w = imggray.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 6
        otsu_result = np.zeros_like(imggray, dtype=np.uint8)
        cv2.circle(otsu_result, (center[1], center[0]), radius, 255, -1)
        brain_cavity = otsu_result.copy()

    print(f"Inicialización: {np.sum(otsu_result > 0)} píxeles activos")

    # Normalizar para morphsnakes (debe estar entre 0 y 1)
    otsu_normalized = otsu_result.astype(np.float64) / 255.0

    # Parámetros adaptativos para morphological_chan_vese
    mean_intensity = np.mean(imggray)
    iterations = 50 if mean_intensity < 0.3 else 30  # Más iteraciones para imágenes oscuras
    smoothing = 2  # Reducir suavizado para conservar más detalles

    print(f"Parámetros MorphACWE: iterations={iterations}, smoothing={smoothing}")

    try:
        # Ejecutar morphological Chan-Vese
        u = ms.morphological_chan_vese(
            imggray,
            iterations,
            init_level_set=otsu_normalized,
            smoothing=smoothing,
            lambda1=1,
            lambda2=1
        )

        print(f"MorphACWE completado. Píxeles finales: {np.sum(u > 0.5)}")

        # Verificar si el resultado es válido
        if np.sum(u > 0.5) == 0:
            print("MorphACWE produjo resultado vacío, usando inicialización")
            u = otsu_normalized

    except Exception as e:
        print(f"Error en MorphACWE: {e}")
        print("Usando inicialización como resultado final")
        u = otsu_normalized

    # Asegurar que u esté en el rango correcto
    u = np.clip(u, 0, 1)

    # Convertir a binario para consistencia
    u_binary = (u > 0.5).astype(np.float64)

    print(f"Resultado final: {np.sum(u_binary)} píxeles segmentados de {u_binary.size} totales")

    # Aplicar post-procesamiento si es necesario
    if np.sum(u_binary) < u_binary.size * 0.001:  # Menos del 0.1%
        print("Resultado muy pequeño, aplicando post-procesamiento...")
        # Usar un círculo más grande
        h, w = imggray.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 3
        u_binary = np.zeros_like(imggray)
        cv2.circle(u_binary, (center[1], center[0]), radius, 1, -1)

    # Invertir si es necesario (pero de manera más robusta)
    try:
        u_final = u_invert(u_binary, imggray)
    except Exception as e:
        print(f"Error en u_invert: {e}")
        u_final = u_binary

    # Dibujar contornos
    try:
        img_contours = draw_contours(u_final, imgcolor if len(imgcolor.shape) == 3 else imggray)
    except Exception as e:
        print(f"Error dibujando contornos: {e}")
        # Crear imagen de contornos básica
        if len(imgcolor.shape) == 3:
            img_contours = (imgcolor * 255).astype(np.uint8)
        else:
            img_contours = cv2.cvtColor((imgcolor * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Asegurar que todas las variables están definidas antes de retornar
    if u_final is None:
        u_final = np.zeros_like(imggray, dtype=np.float64)
    if img_contours is None:
        img_contours = np.zeros((*imggray.shape, 3), dtype=np.uint8)
    if otsu_result is None:
        otsu_result = np.zeros_like(imggray, dtype=np.uint8)
    if brain_cavity is None:
        brain_cavity = np.zeros_like(imggray, dtype=np.uint8)

    return u_final, img_contours, otsu_result, brain_cavity


def simple_brain_segmentation(gray_image):
    """Segmentación simplificada para casos donde fallan otros métodos"""
    print("Usando segmentación simplificada...")

    # Método 1: Otsu directo en toda la imagen
    try:
        otsu_threshold = filters.threshold_otsu(gray_image)
        _, simple_mask = cv2.threshold(gray_image, otsu_threshold, 255, cv2.THRESH_BINARY)

        # Operaciones morfológicas básicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        simple_mask = cv2.morphologyEx(simple_mask, cv2.MORPH_CLOSE, kernel)
        simple_mask = cv2.morphologyEx(simple_mask, cv2.MORPH_OPEN, kernel)

        # Filtrar componentes muy pequeñas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(simple_mask)
        min_size = gray_image.shape[0] * gray_image.shape[1] * 0.01  # 1% del área total

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                simple_mask[labels == i] = 0

        if np.sum(simple_mask > 0) > 0:
            print(f"Segmentación simple exitosa: {np.sum(simple_mask > 0)} píxeles")
            return simple_mask
    except Exception as e:
        print(f"Error en segmentación simple: {e}")

    # Método 2: Umbral adaptativo si Otsu falla
    print("Probando umbral adaptativo...")
    try:
        adaptive_mask = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        adaptive_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_OPEN, kernel)

        if np.sum(adaptive_mask > 0) > 0:
            print(f"Umbral adaptativo exitoso: {np.sum(adaptive_mask > 0)} píxeles")
            return adaptive_mask
    except Exception as e:
        print(f"Error en umbral adaptativo: {e}")

    # Método 3: Segmentación por percentiles
    print("Probando segmentación por percentiles...")
    try:
        threshold_75 = np.percentile(gray_image, 75)
        _, percentile_mask = cv2.threshold(gray_image, threshold_75, 255, cv2.THRESH_BINARY)

        if np.sum(percentile_mask > 0) > 0:
            print(f"Segmentación por percentiles exitosa: {np.sum(percentile_mask > 0)} píxeles")
            return percentile_mask
    except Exception as e:
        print(f"Error en segmentación por percentiles: {e}")

    # Método 4: Círculo central como último recurso
    print("Usando círculo central como último recurso...")
    h, w = gray_image.shape
    center = (h // 2, w // 2)
    radius = min(h, w) // 4
    circle_mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.circle(circle_mask, (center[1], center[0]), radius, 255, -1)

    return circle_mask


def improved_otsu_threshold(image):
    """Otsu mejorado para imágenes con diferentes características de contraste"""
    # Asegurar tipo uint8
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    # Escala de grises si es necesario
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    print(f"Procesando imagen con rango [{gray_image.min()}, {gray_image.max()}]")

    # Mejorar contraste si la imagen es muy oscura
    if np.mean(gray_image) < 80:
        print("Mejorando contraste de imagen oscura...")
        gray_image = enhance_low_contrast_image(gray_image)

    # Método más simple: Otsu directo en toda la imagen
    try:
        otsu_threshold = filters.threshold_otsu(gray_image)
        print(f"Umbral Otsu global: {otsu_threshold}")

        _, otsu_result = cv2.threshold(gray_image, otsu_threshold, 255, cv2.THRESH_BINARY)

        # Operaciones morfológicas básicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        otsu_result = cv2.morphologyEx(otsu_result, cv2.MORPH_OPEN, kernel)
        otsu_result = cv2.morphologyEx(otsu_result, cv2.MORPH_CLOSE, kernel)

        brain_cavity = otsu_result.copy()

        print(f"Otsu global: {np.sum(otsu_result > 0)} píxeles")

        return otsu_result, brain_cavity

    except Exception as e:
        print(f"Error en Otsu mejorado: {e}")
        dummy_shape = image.shape
        return np.zeros(dummy_shape, dtype=np.uint8), np.zeros(dummy_shape, dtype=np.uint8)


# El resto de las funciones permanecen igual...
def rgb2gray(img):
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def enhance_low_contrast_image(image):
    """Mejora el contraste de imágenes oscuras o con bajo contraste"""
    # Normalización adaptativa
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=20)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)

    # Filtro de suavizado para reducir ruido
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return enhanced


def adaptive_skull_detection(gray_image):
    """Detección adaptativa del cráneo para imágenes con diferentes contrastes"""
    # Análisis del histograma para determinar estrategia
    cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    mean_intensity = np.mean(gray_image)
    std_intensity = np.std(gray_image)

    print(f"Intensidad media: {mean_intensity:.2f}, Desviación estándar: {std_intensity:.2f}")

    # Para imágenes muy oscuras
    if mean_intensity < 50:
        # Usar percentil alto en lugar de umbral fijo
        threshold_value = np.percentile(gray_image, 95)
        threshold_value = max(threshold_value, 30)  # Mínimo umbral
        print(f"Imagen oscura detectada. Usando umbral: {threshold_value}")
    elif mean_intensity < 100:
        # Imágenes de contraste medio
        threshold_value = np.percentile(gray_image, 85)  # Reducido de 90 a 85
        threshold_value = max(threshold_value, 60)  # Reducido de 80 a 60
        print(f"Imagen de contraste medio. Usando umbral: {threshold_value}")
    else:
        # Imágenes con buen contraste
        threshold_value = max(200, np.percentile(gray_image, 85))
        print(f"Imagen de buen contraste. Usando umbral: {threshold_value}")

    # Aplicar umbralización
    _, skull_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Operaciones morfológicas adaptativas
    kernel_size = max(5, int(min(gray_image.shape) * 0.02))  # Tamaño adaptativo
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Cerrado para conectar estructuras del cráneo
    skull_mask = cv2.morphologyEx(skull_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Rellenado de huecos
    skull_mask = ndimage.binary_fill_holes(skull_mask).astype(np.uint8) * 255

    return skull_mask, threshold_value


def robust_brain_cavity_extraction(skull_mask, gray_image):
    """Extracción robusta de la cavidad craneal"""
    # Encontrar contornos
    contours, _ = cv2.findContours(skull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No se encontraron contornos del cráneo")
        return np.zeros_like(gray_image), None

    # Filtrar contornos por área mínima (reducir el umbral)
    min_area = (gray_image.shape[0] * gray_image.shape[1]) * 0.05  # Reducido de 0.1 a 0.05
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        print(f"No se encontraron contornos válidos con área > {min_area}")
        # Intentar con el contorno más grande disponible
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            print(f"Usando el contorno más grande disponible con área: {largest_area}")
            valid_contours = [largest_contour]
        else:
            return np.zeros_like(gray_image), None

    # Seleccionar el contorno más circular (más parecido al cráneo)
    best_contour = None
    best_score = 0

    for contour in valid_contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > best_score:
                best_score = circularity
                best_contour = contour

    if best_contour is None:
        best_contour = max(valid_contours, key=cv2.contourArea)

    print(f"Circularidad del contorno seleccionado: {best_score:.3f}")

    # Crear máscara del cráneo
    skull_filled = np.zeros_like(gray_image)
    cv2.drawContours(skull_filled, [best_contour], -1, 255, thickness=cv2.FILLED)

    # Erosión adaptativa para obtener la cavidad craneal
    image_size = min(gray_image.shape)
    erosion_size = max(3, int(image_size * 0.015))  # Erosión adaptativa
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))

    brain_cavity = cv2.erode(skull_filled, kernel_erode, iterations=2)

    # Asegurar que la cavidad no sea demasiado pequeña
    cavity_area = np.sum(brain_cavity > 0)
    skull_area = np.sum(skull_filled > 0)

    if cavity_area < skull_area * 0.3:  # Si la cavidad es menor al 30% del cráneo
        print("Cavidad muy pequeña, ajustando erosión...")
        erosion_size = max(2, erosion_size // 2)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        brain_cavity = cv2.erode(skull_filled, kernel_erode, iterations=1)

    return brain_cavity, best_contour


def u_invert(u, imggray):
    """Función mejorada para invertir segmentación si es necesario"""
    try:
        u_uint8 = (u * 255).astype(np.uint8) if u.max() <= 1.0 else u.astype(np.uint8)

        contours, _ = cv2.findContours(u_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print("No se encontraron contornos para inversión")
            return u.astype(np.float64)

        height, width = imggray.shape
        # Usar menos puntos aleatorios para mayor estabilidad
        num_points = min(50, height * width // 100)
        random_h = np.random.randint(0, height, size=num_points)
        random_w = np.random.randint(0, width, size=num_points)
        coordinates = np.column_stack((random_h, random_w))

        pixel_inside = []
        pixel_outside = []
        point_inside = []
        point_outside = []

        for (y, x) in coordinates:
            point = (np.float32(x), np.float32(y))
            is_inside_any = False

            for contour in contours:
                is_inside = cv2.pointPolygonTest(contour, point, False)
                if is_inside >= 0:  # Incluir puntos en el borde
                    pixel_inside.append(imggray[y, x])
                    point_inside.append((y, x))
                    is_inside_any = True
                    break

            if not is_inside_any:
                pixel_outside.append(imggray[y, x])
                point_outside.append((y, x))

        if len(pixel_inside) == 0 or len(pixel_outside) == 0:
            print("No hay suficientes puntos para análisis de inversión")
            return u.astype(np.float64)

        mean_pixel_inside = np.mean(pixel_inside)
        mean_pixel_outside = np.mean(pixel_outside)

        print(f"Media dentro: {mean_pixel_inside:.3f}, Media fuera: {mean_pixel_outside:.3f}")

        # Si la media dentro es menor que fuera, puede que necesitemos invertir
        if mean_pixel_inside < mean_pixel_outside:
            print("Invirtiendo segmentación...")
            u_result = 1.0 - u.astype(np.float64)
        else:
            u_result = u.astype(np.float64)

        return u_result

    except Exception as e:
        print(f"Error en u_invert: {e}")
        return u.astype(np.float64)


def draw_contours(u, imgcolor):
    """Función mejorada para dibujar contornos"""
    try:
        img_u = (u * 255).astype(np.uint8) if u.max() <= 1.0 else u.astype(np.uint8)
        contours, _ = cv2.findContours(img_u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(imgcolor.shape) == 3:
            img_contours = imgcolor.copy()
            if img_contours.max() <= 1.0:
                img_contours = (img_contours * 255).astype(np.uint8)
            img_contours = cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 1)
        else:
            img_contours = cv2.cvtColor((imgcolor * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img_contours = cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 1)

        return img_contours

    except Exception as e:
        print(f"Error dibujando contornos: {e}")
        # Retornar imagen original si hay error
        if len(imgcolor.shape) == 3:
            return (imgcolor * 255).astype(np.uint8)
        else:
            return cv2.cvtColor((imgcolor * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def coverage_rate(levelset):
    """Calcula la tasa de cobertura"""
    if levelset is None or levelset.size == 0:
        return 0.0

    rows, cols = levelset.shape
    sum_pixel = rows * cols
    white_pixel = np.sum(levelset > 0.5) if levelset.max() <= 1.0 else np.sum(levelset > 127)
    rate = (white_pixel / sum_pixel) * 100
    return rate


def uniformity_rate(levelset, grid_size):
    """Calcula la tasa de uniformidad mejorada"""
    if levelset is None or levelset.size == 0:
        print("Levelset vacío para uniformity_rate")
        return 0.0

    rows, cols = levelset.shape
    num_rows = rows // grid_size
    num_cols = cols // grid_size

    # Validar que hay suficiente espacio para hacer grids
    if num_rows == 0 or num_cols == 0:
        print(f"Imagen demasiado pequeña para grid_size {grid_size}")
        return 0.0

    grid_image = []

    for i in range(num_rows):
        for j in range(num_cols):
            left = j * grid_size
            top = i * grid_size
            right = left + grid_size
            bottom = top + grid_size
            grid_image.append(levelset[top:bottom, left:right])

    grid_rate = []
    for grid in grid_image:
        rate = coverage_rate(grid)
        grid_rate.append(rate)

    if len(grid_rate) == 0:
        return 0.0

    grid_rate_sum = sum(grid_rate)

    # Validar división por cero
    if grid_rate_sum == 0:
        print("Suma de grid_rate es 0")
        return 0.0

    grid_rate_mean = grid_rate_sum / len(grid_rate)

    tmp = sum(abs(rate - grid_rate_mean) for rate in grid_rate)

    res = 100 - (tmp / grid_rate_sum) * 100

    # Asegurar que el resultado esté en rango válido
    res = max(0.0, min(100.0, res))

    # Verificar si es NaN
    if np.isnan(res):
        print("Resultado NaN en uniformity_rate, retornando 0")
        return 0.0

    return res
