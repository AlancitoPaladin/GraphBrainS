from utils import test_MorphACWE, coverage_rate, uniformity_rate


class Imgseg():
    def __init__(self, img_path, grid_size):
        self.img_path = img_path
        self.grid_size = grid_size
        # Detecta automáticamente si es DICOM
        if isinstance(img_path, str):
            self.is_dicom = img_path.lower().endswith('.dcm')
        else:
            self.is_dicom = False  # Es un array de imagen

    def seg_rate(self):
        # Usar la misma función para todos los tipos de imagen
        u, img_contours, otsu_result, brain_cavity = test_MorphACWE(self.img_path)
        return u, img_contours, coverage_rate(u), uniformity_rate(u, self.grid_size), otsu_result, brain_cavity
