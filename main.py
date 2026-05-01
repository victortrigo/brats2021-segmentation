import argparse
import shutil
import os

# Importamos los orquestadores desde el paquete src
from src.run_hpo import run_hpo
from src.run_phase2_factorial import main as run_phase2
from src.run_phase3_evaluate import main as run_phase3


def reset_experiments():
    """
    Deletes all experiment directories to start from a clean slate.
    Ensures no leftover YAMLs or .pth files contaminate new runs.
    """
    print("\n" + "!"*50)
    print("🧹 RESET MODE ACTIVATED: Cleaning previous experiments...")
    
    # Lista con las 3 fases del proyecto
    dirs_to_clean = [
        "experiment/phase1_hpo", 
        "experiment/phase2_factorial",
        "experiment/phase3_evaluation"
    ]
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            # shutil.rmtree elimina la carpeta y todo su contenido
            shutil.rmtree(d) 
            print(f"  🗑️  Deleted: {d}")
    
    # Volvemos a crear las carpetas vacías para mantener la estructura
    for d in dirs_to_clean:
        os.makedirs(d, exist_ok=True)
    
    print("✨ Environment is clean and ready for training.")
    print("!"*50 + "\n")


def main():
    """
    Main entry point for the BraTS 2021 Experiment Pipeline.
    Orchestrates HPO, Factorial Training, and Clinical Evaluation.
    """
    parser = argparse.ArgumentParser(description="BraTS 2021 Experiments Pipeline")
    
    # Menú de opciones principal
    parser.add_argument('--mode', type=str, required=True, choices=['hpo', 'phase2'],
                        help="Choose 'hpo', 'phase2', 'phase3', or 'reset'")
    
    # Flag para controlar la aceleración de hardware (Mixed Precision)
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help="Disable 16-bit acceleration (Mixed Precision)")

    args = parser.parse_args()

    # Si elegimos limpiar, lo hacemos y terminamos el programa aquí
    if args.mode == 'reset':
        reset_experiments()
        return

    print(f"Starting in mode: {args.mode.upper()}")
    print(f"AMP Acceleration (16-bit): {'ENABLED' if args.use_amp else 'DISABLED'}")

    # ==========================================================
    # ENRUTADOR DE FASES
    # ==========================================================
    if args.mode == 'hpo':
        # Red de seguridad: nos pregunta si queremos borrar todo antes de buscar hiperparámetros
        respuesta = input("Do you want to clean previous experiments before starting HPO? (y/n): ")
        if respuesta.lower() == 'y':
            reset_experiments()
        run_hpo() 

    elif args.mode == 'phase2':
        run_phase2()

    elif args.mode == 'phase3':
        run_phase3()

if __name__ == "__main__":
    main()
