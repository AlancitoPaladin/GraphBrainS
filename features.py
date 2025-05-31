import os

import cv2
import numpy as np
import pandas as pd
from skimage import measure


def _extract_morphological_features(u_segmentation):
    """Extrae características morfológicas básicas"""
    features = {}

    # Convertir a binario
    if u_segmentation.max() <= 1.0:
        binary_mask = (u_segmentation > 0.5).astype(np.uint8)
    else:
        binary_mask = (u_segmentation > 127).astype(np.uint8)

    # Área total
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    tumor_pixels = np.sum(binary_mask > 0)
    features['tumor_area_pixels'] = tumor_pixels
    features['tumor_area_percentage'] = (tumor_pixels / total_pixels) * 100

    # Encontrar contornos
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)

        # Perímetro
        perimeter = cv2.arcLength(largest_contour, True)
        features['perimeter'] = perimeter

        # Compacidad (circularidad)
        if perimeter > 0:
            compactness = 4 * np.pi * tumor_pixels / (perimeter ** 2)
            features['compactness'] = compactness
        else:
            features['compactness'] = 0

        # Bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        features['bbox_width'] = w
        features['bbox_height'] = h
        features['bbox_area'] = w * h
        features['bbox_aspect_ratio'] = w / h if h > 0 else 0

        # Extent (qué tan lleno está el bounding box)
        features['extent'] = tumor_pixels / (w * h) if (w * h) > 0 else 0

        # Número de contornos (indica fragmentación)
        features['num_components'] = len(contours)

    else:
        # Valores por defecto si no hay contornos
        for key in ['perimeter', 'compactness', 'bbox_width', 'bbox_height',
                    'bbox_area', 'bbox_aspect_ratio', 'extent', 'num_components']:
            features[key] = 0

    return features


class TumorCharacterizer:
    """
    Caracterizador de tumores basado en las métricas existentes (Coverage y Uniformity)
    y características adicionales derivadas de la segmentación MorphACWE
    """

    def __init__(self):
        self.tumor_patterns = {
            'normal': {'coverage_range': (0, 5), 'uniformity_range': (0, 30)},
            'meningioma': {'coverage_range': (5, 25), 'uniformity_range': (60, 90)},
            'glioma': {'coverage_range': (15, 60), 'uniformity_range': (20, 70)},
            'tumor_pituitario': {'coverage_range': (1, 8), 'uniformity_range': (70, 95)}
        }

    def extract_enhanced_features(self, u_segmentation, img_original, coverage_rate, uniformity_rate,
                                  otsu_result=None, brain_cavity=None, grid_size=80):
        """
        Extrae características mejoradas basándose en los resultados existentes
        """
        features = {'coverage_rate': coverage_rate, 'uniformity_rate': uniformity_rate, 'grid_size': grid_size}

        # Métricas base existentes

        # Características morfológicas básicas
        morph_features = _extract_morphological_features(u_segmentation)
        features.update(morph_features)

        # Características de intensidad
        intensity_features = self._extract_intensity_features(u_segmentation, img_original)
        features.update(intensity_features)

        # Características de distribución espacial
        spatial_features = self._extract_spatial_features(u_segmentation, grid_size)
        features.update(spatial_features)

        # Características de calidad de segmentación
        quality_features = self._extract_quality_features(u_segmentation, otsu_result, brain_cavity)
        features.update(quality_features)

        # Características derivadas (ratios y combinaciones)
        derived_features = self._extract_derived_features(features)
        features.update(derived_features)

        return features

    @staticmethod
    def _extract_intensity_features(u_segmentation, img_original):
        """Extrae características de intensidad"""
        features = {}

        # Convertir imagen a escala de grises si es necesario
        if len(img_original.shape) == 3:
            gray_img = np.mean(img_original, axis=2)
        else:
            gray_img = img_original.copy()

        # Normalizar a [0, 255]
        if gray_img.max() <= 1.0:
            gray_img = gray_img * 255

        # Máscara binaria
        if u_segmentation.max() <= 1.0:
            mask = (u_segmentation > 0.5).astype(bool)
        else:
            mask = (u_segmentation > 127).astype(bool)

        if np.sum(mask) > 0:
            # Intensidades en la región del tumor
            tumor_intensities = gray_img[mask]

            features['mean_intensity'] = np.mean(tumor_intensities)
            features['std_intensity'] = np.std(tumor_intensities)
            features['min_intensity'] = np.min(tumor_intensities)
            features['max_intensity'] = np.max(tumor_intensities)
            features['intensity_range'] = features['max_intensity'] - features['min_intensity']

            # Percentiles
            features['intensity_p25'] = np.percentile(tumor_intensities, 25)
            features['intensity_p50'] = np.percentile(tumor_intensities, 50)
            features['intensity_p75'] = np.percentile(tumor_intensities, 75)

            # Contraste con región circundante
            # Dilatar la máscara para obtener región circundante
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            surrounding_mask = dilated_mask & (~mask)

            if np.sum(surrounding_mask) > 0:
                surrounding_intensities = gray_img[surrounding_mask]
                features['surrounding_mean_intensity'] = np.mean(surrounding_intensities)
                features['tumor_surrounding_contrast'] = abs(
                    features['mean_intensity'] - features['surrounding_mean_intensity'])
                features['tumor_surrounding_ratio'] = features['mean_intensity'] / features[
                    'surrounding_mean_intensity'] if features['surrounding_mean_intensity'] > 0 else 1
            else:
                features['surrounding_mean_intensity'] = 0
                features['tumor_surrounding_contrast'] = 0
                features['tumor_surrounding_ratio'] = 1

            # Heterogeneidad interna del tumor
            features['intensity_coefficient_variation'] = features['std_intensity'] / features['mean_intensity'] if \
                features['mean_intensity'] > 0 else 0

        else:
            # Valores por defecto si no hay tumor
            for key in ['mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
                        'intensity_range', 'intensity_p25', 'intensity_p50', 'intensity_p75',
                        'surrounding_mean_intensity', 'tumor_surrounding_contrast',
                        'tumor_surrounding_ratio', 'intensity_coefficient_variation']:
                features[key] = 0

        return features

    @staticmethod
    def _extract_spatial_features(u_segmentation, grid_size):
        """Extrae características de distribución espacial"""
        features = {}

        # Convertir a binario
        if u_segmentation.max() <= 1.0:
            binary_mask = (u_segmentation > 0.5).astype(np.uint8)
        else:
            binary_mask = (u_segmentation > 127).astype(np.uint8)

        rows, cols = binary_mask.shape

        # Centro de masa
        if np.sum(binary_mask) > 0:
            moments = cv2.moments(binary_mask)
            if moments['m00'] != 0:
                centroid_x = moments['m10'] / moments['m00']
                centroid_y = moments['m01'] / moments['m00']

                # Normalizar por tamaño de imagen
                features['centroid_x_normalized'] = centroid_x / cols
                features['centroid_y_normalized'] = centroid_y / rows

                # Distancia al centro de la imagen
                center_x, center_y = cols / 2, rows / 2
                distance_to_center = np.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)
                max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
                features['distance_to_center_normalized'] = distance_to_center / max_distance

                # Cuadrante
                quadrant = 1
                if centroid_x >= center_x and centroid_y < center_y:
                    quadrant = 2
                elif centroid_x < center_x and centroid_y >= center_y:
                    quadrant = 3
                elif centroid_x >= center_x and centroid_y >= center_y:
                    quadrant = 4
                features['quadrant'] = quadrant
            else:
                features['centroid_x_normalized'] = 0.5
                features['centroid_y_normalized'] = 0.5
                features['distance_to_center_normalized'] = 0
                features['quadrant'] = 0
        else:
            features['centroid_x_normalized'] = 0.5
            features['centroid_y_normalized'] = 0.5
            features['distance_to_center_normalized'] = 0
            features['quadrant'] = 0

        # Distribución por grids (usando el grid_size existente)
        num_rows = rows // grid_size
        num_cols = cols // grid_size

        if num_rows > 0 and num_cols > 0:
            grid_coverages = []
            for i in range(num_rows):
                for j in range(num_cols):
                    grid_region = binary_mask[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]
                    grid_coverage = np.sum(grid_region > 0) / (grid_size * grid_size) * 100
                    grid_coverages.append(grid_coverage)

            features['grid_coverage_mean'] = np.mean(grid_coverages)
            features['grid_coverage_std'] = np.std(grid_coverages)
            features['grid_coverage_max'] = np.max(grid_coverages)
            features['grids_with_tumor'] = np.sum(np.array(grid_coverages) > 0)
            features['total_grids'] = len(grid_coverages)
            features['grid_occupation_ratio'] = features['grids_with_tumor'] / features['total_grids']
        else:
            for key in ['grid_coverage_mean', 'grid_coverage_std', 'grid_coverage_max',
                        'grids_with_tumor', 'total_grids', 'grid_occupation_ratio']:
                features[key] = 0

        return features

    @staticmethod
    def _extract_quality_features(u_segmentation, otsu_result, brain_cavity):
        """Extrae características de calidad de la segmentación"""
        features = {}

        # Convertir a binario
        if u_segmentation.max() <= 1.0:
            final_mask = (u_segmentation > 0.5).astype(np.uint8)
        else:
            final_mask = (u_segmentation > 127).astype(np.uint8)

        # Comparación con Otsu inicial
        if otsu_result is not None:
            otsu_binary = (otsu_result > 127).astype(np.uint8)

            # Similaridad con Otsu (Índice de Jaccard)
            intersection = np.sum((final_mask > 0) & (otsu_binary > 0))
            union = np.sum((final_mask > 0) | (otsu_binary > 0))
            features['jaccard_with_otsu'] = intersection / union if union > 0 else 0

            # Cambio de área desde Otsu
            otsu_area = np.sum(otsu_binary > 0)
            final_area = np.sum(final_mask > 0)
            features['area_change_from_otsu'] = (final_area - otsu_area) / otsu_area if otsu_area > 0 else 0
        else:
            features['jaccard_with_otsu'] = 0
            features['area_change_from_otsu'] = 0

        # Comparación con cavidad craneal
        if brain_cavity is not None:
            cavity_binary = (brain_cavity > 127).astype(np.uint8)

            # Proporción del tumor respecto a la cavidad craneal
            cavity_area = np.sum(cavity_binary > 0)
            tumor_area = np.sum(final_mask > 0)
            features['tumor_to_brain_ratio'] = tumor_area / cavity_area if cavity_area > 0 else 0

            # Tumor fuera de la cavidad craneal (indica problema de segmentación)
            tumor_outside_brain = np.sum((final_mask > 0) & (cavity_binary == 0))
            features['tumor_outside_brain_ratio'] = tumor_outside_brain / tumor_area if tumor_area > 0 else 0
        else:
            features['tumor_to_brain_ratio'] = 0
            features['tumor_outside_brain_ratio'] = 0

        # Conectividad del tumor
        labeled_mask = measure.label(final_mask)
        num_components = labeled_mask.max()
        features['connectivity_components'] = num_components

        if num_components > 0:
            # Tamaño del componente más grande vs total
            component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            largest_component_size = max(component_sizes)
            total_tumor_size = sum(component_sizes)
            features['largest_component_ratio'] = largest_component_size / total_tumor_size

            # Fragmentación (número de componentes pequeños)
            small_components = sum(1 for size in component_sizes if size < total_tumor_size * 0.1)
            features['small_components_count'] = small_components
        else:
            features['largest_component_ratio'] = 0
            features['small_components_count'] = 0

        return features

    @staticmethod
    def _extract_derived_features(features):
        """Extrae características derivadas y ratios"""
        derived = {}

        # Índice de regularidad (basado en coverage y uniformity)
        coverage = features.get('coverage_rate', 0)
        uniformity = features.get('uniformity_rate', 0)

        # Índice combinado de regularidad
        derived['regularity_index'] = (uniformity * coverage) / 100 if coverage > 0 else 0

        # Clasificación de tamaño
        if coverage < 5:
            derived['size_category'] = 0  # Muy pequeño/Normal
        elif coverage < 15:
            derived['size_category'] = 1  # Pequeño
        elif coverage < 30:
            derived['size_category'] = 2  # Mediano
        else:
            derived['size_category'] = 3  # Grande

        # Clasificación de uniformidad
        if uniformity < 30:
            derived['uniformity_category'] = 0  # Muy irregular
        elif uniformity < 60:
            derived['uniformity_category'] = 1  # Irregular
        elif uniformity < 80:
            derived['uniformity_category'] = 2  # Regular
        else:
            derived['uniformity_category'] = 3  # Muy regular

        # Ratio de aspecto normalizado
        aspect_ratio = features.get('bbox_aspect_ratio', 1)
        derived['aspect_ratio_deviation'] = abs(aspect_ratio - 1.0)  # Desviación de la forma cuadrada

        # Índice de densidad (coverage vs área del bounding box)
        bbox_area = features.get('bbox_area', 1)
        tumor_area = features.get('tumor_area_pixels', 0)
        features.get('tumor_area_pixels', 0) / (
                features.get('tumor_area_percentage', 1) / 100) if features.get('tumor_area_percentage',
                                                                                0) > 0 else 1
        derived['density_index'] = tumor_area / bbox_area if bbox_area > 0 else 0

        # Índice de complejidad (combina múltiples factores)
        compactness = features.get('compactness', 0)
        num_components = features.get('connectivity_components', 1)
        derived['complexity_index'] = (1 - compactness) * num_components

        return derived

    @staticmethod
    def classify_tumor_pattern(features):
        """
        Clasifica el patrón del tumor basándose en coverage y uniformity principalmente
        """
        coverage = features.get('coverage_rate', 0)
        uniformity = features.get('uniformity_rate', 0)
        regularity = features.get('regularity_index', 0)
        features.get('size_category', 0)
        tumor_to_brain = features.get('tumor_to_brain_ratio', 0)

        # Reglas de clasificación basadas en patrones observados
        if coverage < 2 and uniformity < 50:
            return "Normal/Sin tumor significativo"
        elif coverage < 10 and uniformity > 70 and regularity > 30:
            return "Meningioma (pequeño, bien definido)"
        elif coverage > 20 and uniformity < 50 and features.get('complexity_index', 0) > 0.5:
            return "Glioma (grande, irregular)"
        elif coverage < 8 and uniformity > 80 and tumor_to_brain < 0.1:
            return "Tumor pituitario (pequeño, central)"
        elif coverage > 10 and 50 <= uniformity <= 75:
            return "Tumor intermedio (requiere análisis adicional)"
        elif uniformity < 30:
            return "Lesión irregular (posible maligna)"
        else:
            return "Patrón no clasificado"

    @staticmethod
    def generate_clinical_report(features, tumor_classification):
        """
        Genera un reporte clínico basado en las características extraídas
        """
        report = ["=" * 60, "REPORTE DE ANÁLISIS DE TUMOR - BASADO EN SEGMENTACIÓN", "=" * 60,
                  f"\nClasificación detectada: {tumor_classification}", f"\n{'MÉTRICAS PRINCIPALES':-^60}",
                  f"Coverage Rate: {features.get('coverage_rate', 0):.2f}%",
                  f"Uniformity Rate: {features.get('uniformity_rate', 0):.2f}%",
                  f"Índice de Regularidad: {features.get('regularity_index', 0):.2f}",
                  f"\n{'CARACTERÍSTICAS MORFOLÓGICAS':-^60}",
                  f"Área del tumor: {features.get('tumor_area_pixels', 0):.0f} píxeles ({features.get('tumor_area_percentage', 0):.2f}%)",
                  f"Compacidad (circularidad): {features.get('compactness', 0):.3f}",
                  f"Relación de aspecto: {features.get('bbox_aspect_ratio', 0):.2f}",
                  f"Número de componentes: {features.get('connectivity_components', 0)}",
                  f"\n{'CARACTERÍSTICAS DE INTENSIDAD':-^60}",
                  f"Intensidad media: {features.get('mean_intensity', 0):.1f}",
                  f"Contraste con tejido circundante: {features.get('tumor_surrounding_contrast', 0):.1f}",
                  f"Heterogeneidad interna: {features.get('intensity_coefficient_variation', 0):.3f}",
                  f"\n{'LOCALIZACIÓN':-^60}",
                  f"Centroide normalizado: ({features.get('centroid_x_normalized', 0):.3f}, {features.get('centroid_y_normalized', 0):.3f})",
                  f"Distancia al centro: {features.get('distance_to_center_normalized', 0):.3f}",
                  f"Cuadrante: {features.get('quadrant', 0)}", f"\n{'DISTRIBUCIÓN ESPACIAL':-^60}",
                  f"Grids ocupados: {features.get('grids_with_tumor', 0)}/{features.get('total_grids', 0)}",
                  f"Ratio de ocupación: {features.get('grid_occupation_ratio', 0):.3f}",
                  f"Coverage máximo por grid: {features.get('grid_coverage_max', 0):.1f}%",
                  f"\n{'CALIDAD DE SEGMENTACIÓN':-^60}",
                  f"Similaridad con Otsu inicial: {features.get('jaccard_with_otsu', 0):.3f}",
                  f"Ratio tumor/cerebro: {features.get('tumor_to_brain_ratio', 0):.3f}",
                  f"Tumor fuera del cerebro: {features.get('tumor_outside_brain_ratio', 0):.3f}",
                  f"\n{'INTERPRETACIÓN CLÍNICA':-^60}"]

        # Información básica

        # Características morfológicas

        # Características de intensidad

        # Localización

        # Distribución espacial

        # Calidad de segmentación

        # Interpretación clínica

        coverage = features.get('coverage_rate', 0)
        uniformity = features.get('uniformity_rate', 0)

        if "Normal" in tumor_classification:
            report.append("• No se detecta evidencia significativa de tumor")
            report.append("• La segmentación puede corresponder a variaciones normales")
        elif "Meningioma" in tumor_classification:
            report.append("• Patrón compatible con meningioma")
            report.append("• Lesión bien definida y regular")
            report.append("• Típicamente benigno, pero requiere confirmación")
        elif "Glioma" in tumor_classification:
            report.append("• Patrón compatible con glioma")
            report.append("• Lesión irregular con bordes mal definidos")
            report.append("• Requiere evaluación urgente por neurólogo")
        elif "pituitario" in tumor_classification:
            report.append("• Patrón compatible con adenoma pituitario")
            report.append("• Lesión pequeña y localizada")
            report.append("• Generalmente benigno")
        else:
            report.append("• Patrón no concluyente")
            report.append("• Se requiere análisis adicional")

        # Recomendaciones
        report.append(f"\n{'RECOMENDACIONES':-^60}")
        if coverage > 20:
            report.append("• Lesión de tamaño considerable, evaluación prioritaria")
        if uniformity < 30:
            report.append("• Baja uniformidad sugiere malignidad, requiere biopsia")
        if features.get('tumor_outside_brain_ratio', 0) > 0.1:
            report.append("• Posible error de segmentación, revisar imagen")

        report.append("• Este análisis es orientativo únicamente")
        report.append("• Validación requerida por especialista en neuroimagen")
        report.append("=" * 60)

        return "\n".join(report)

    @staticmethod
    def save_features_csv(features, filename):
        """Guarda las características en CSV"""
        df = pd.DataFrame([features])
        df.to_csv(filename, index=False)
        print(f"Características guardadas en: {filename}")


# Función para integrar con el código existente
def analyze_tumor_from_results(u_segmentation, coverage_rate, uniformity_rate,
                               otsu_result, brain_cavity, img_original, output_dir, base_name, grid_size=80):
    """
    Función para integrar el análisis de tumor con los resultados existentes
    """
    characterizer = TumorCharacterizer()

    # Extraer características mejoradas
    features = characterizer.extract_enhanced_features(
        u_segmentation, img_original, coverage_rate, uniformity_rate,
        otsu_result, brain_cavity, grid_size
    )

    # Clasificar patrón del tumor
    tumor_classification = characterizer.classify_tumor_pattern(features)

    # Generar reporte clínico
    clinical_report = characterizer.generate_clinical_report(features, tumor_classification)

    # Guardar archivos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar reporte clínico
    report_path = os.path.join(output_dir, f"{base_name}_clinical_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(clinical_report)

    # Guardar características en CSV
    features_path = os.path.join(output_dir, f"{base_name}_detailed_features.csv")
    characterizer.save_features_csv(features, features_path)

    print(f"\n{'=' * 60}")
    print("ANÁLISIS CLÍNICO COMPLETADO")
    print(f"{'=' * 60}")
    print(f"Clasificación: {tumor_classification}")
    print(f"Coverage: {coverage_rate:.2f}% | Uniformity: {uniformity_rate:.2f}%")
    print(f"Reporte guardado en: {report_path}")
    print(f"Características en: {features_path}")

    return {
        'features': features,
        'classification': tumor_classification,
        'clinical_report': clinical_report
    }


if __name__ == "__main__":
    print("Módulo de caracterización de tumores cargado")
    print("Uso: analyze_tumor_from_results(...)")
