import hashlib
import os
from pathlib import Path

# Configuración de rutas
PROJECT_DIR = Path(".")  # Directorio actual del proyecto
IMAGES_DIR = PROJECT_DIR / "images1" / "1"
LABELS = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]  # Ajusta según tus etiquetas


def compute_hash(file):
    """Calcula el hash MD5 de un archivo"""
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def list_files(hash_dict):
    """Recorre las carpetas Training y Testing y calcula hashes de todas las imágenes"""
    for data_type in ['Training', 'Testing']:
        data_path = IMAGES_DIR / data_type
        
        if not data_path.exists():
            print(f"Advertencia: La carpeta {data_path} no existe")
            continue
            
        print(f"Procesando carpeta: {data_path}")
        
        # Si hay subcarpetas por etiquetas
        if any((data_path / label).exists() for label in LABELS):
            for label in LABELS:
                folder_path = data_path / label
                if folder_path.exists():
                    process_folder(folder_path, hash_dict)
                else:
                    print(f"Carpeta no encontrada: {folder_path}")
        else:
            # Si no hay subcarpetas, procesar directamente
            process_folder(data_path, hash_dict)


def process_folder(folder_path, hash_dict):
    """Procesa una carpeta específica buscando archivos de imagen"""
    print(f"Escaneando: {folder_path}")
    file_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Buscar múltiples formatos de imagen
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm')):
                file_path = os.path.join(root, file)
                file_hash = compute_hash(file_path)
                
                if file_hash in hash_dict:
                    hash_dict[file_hash].append(file_path)
                else:
                    hash_dict[file_hash] = [file_path]
                
                file_count += 1
    
    print(f"  -> Archivos procesados: {file_count}")


def remove_duplicates(hash_dict):
    """Elimina archivos duplicados basándose en el hash"""
    duplicate_count = 0
    total_duplicates = 0
    
    print("\n=== ANÁLISIS DE DUPLICADOS ===")
    
    for hash_value, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            print(f"\nDuplicados encontrados (hash: {hash_value[:8]}...):")
            for i, file_path in enumerate(file_paths):
                print(f"  {i+1}. {file_path}")
            
            # Conservar el primer archivo, eliminar el resto
            for file_path in file_paths[1:]:
                try:
                    print(f"  -> Eliminando: {file_path}")
                    os.remove(file_path)
                    duplicate_count += 1
                except OSError as e:
                    print(f"  -> Error al eliminar {file_path}: {e}")
            
            total_duplicates += len(file_paths) - 1
    
    print(f"\n=== RESUMEN ===")
    print(f"Archivos duplicados eliminados: {duplicate_count}")
    print(f"Grupos de duplicados encontrados: {len([files for files in hash_dict.values() if len(files) > 1])}")


def analyze_dataset():
    """Analiza el dataset antes de la limpieza"""
    print("=== ANÁLISIS DEL DATASET ===")
    
    for data_type in ['Training', 'Testing']:
        data_path = IMAGES_DIR / data_type
        
        if not data_path.exists():
            print(f"Carpeta no encontrada: {data_path}")
            continue
            
        print(f"\n{data_type}:")
        
        # Si hay subcarpetas por etiquetas
        if any((data_path / label).exists() for label in LABELS):
            for label in LABELS:
                folder_path = data_path / label
                if folder_path.exists():
                    count = count_images_in_folder(folder_path)
                    print(f"  {label}: {count} imágenes")
                else:
                    print(f"  {label}: Carpeta no encontrada")
        else:
            # Si no hay subcarpetas
            count = count_images_in_folder(data_path)
            print(f"  Total: {count} imágenes")


def count_images_in_folder(folder_path):
    """Cuenta imágenes en una carpeta"""
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm')):
                count += 1
    return count


def verify_structure():
    """Verifica la estructura de carpetas"""
    print("=== VERIFICACIÓN DE ESTRUCTURA ===")
    print(f"Directorio base: {IMAGES_DIR}")
    
    if not IMAGES_DIR.exists():
        print(f"ERROR: El directorio {IMAGES_DIR} no existe")
        return False
    
    for data_type in ['Training', 'Testing']:
        data_path = IMAGES_DIR / data_type
        exists = data_path.exists()
        print(f"{data_type}: {'✓' if exists else '✗'} {data_path}")
        
        if exists:
            # Verificar si hay subcarpetas
            subdirs = [d for d in data_path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"  Subcarpetas encontradas: {[d.name for d in subdirs]}")
            else:
                print("  Sin subcarpetas (archivos directamente en la carpeta)")
    
    return True


if __name__ == '__main__':
    print("LIMPIEZA DE DATASET - DETECCIÓN DE DUPLICADOS")
    print("=" * 50)
    
    # Verificar estructura
    if not verify_structure():
        exit(1)
    
    # Analizar dataset
    analyze_dataset()
    
    # Preguntar confirmación
    response = input("\n¿Continuar con la detección y eliminación de duplicados? (s/n): ")
    if response.lower() not in ['s', 'si', 'y', 'yes']:
        print("Operación cancelada.")
        exit(0)
    
    # Procesar duplicados
    print("\n=== PROCESANDO DUPLICADOS ===")
    hash_dict = {}
    list_files(hash_dict)
    
    print(f"Total de archivos únicos analizados: {len(hash_dict)}")
    
    if hash_dict:
        remove_duplicates(hash_dict)
        
        # Análisis final
        print("\n=== ANÁLISIS POST-LIMPIEZA ===")
        analyze_dataset()
    else:
        print("No se encontraron archivos para procesar.")