import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.abspath('src'))

class TestPhase3Evaluate(unittest.TestCase):
    """Test suite for Phase 3 (Clinical Evaluation Orchestrator)."""

    @patch('src.run_phase3_evaluate.DataLoader')
    @patch('src.run_phase3_evaluate.Dataset')
    @patch('src.run_phase3_evaluate.evaluate_model')
    @patch('src.run_phase3_evaluate.get_model')
    @patch('src.run_phase3_evaluate.torch.load')
    def test_fase3_orchestrator(self, mock_torch_load, mock_get_model, mock_evaluate_model, mock_dataset, mock_dataloader):
        print("\n" + "="*50)
        print("🧪 STARTING TEST: PHASE 3 (Evaluation Logic)")
        print("="*50)

        # 1. Configurar los Mocks de PyTorch
        mock_dataset.return_value = MagicMock()
        mock_dataloader.return_value = []
        
        # Simulamos una red neuronal vacía
        mock_model_instance = MagicMock()
        mock_get_model.return_value.to.return_value = mock_model_instance
        mock_torch_load.return_value = {} # Pesos falsos
        
        # Simulamos que tu función evaluate_model retorna el diccionario clínico por cada clase
        mock_evaluate_model.return_value = {
            'Dice_NCR': 0.85, 'Dice_ED': 0.78, 'Dice_ET': 0.82,
            'HD95_NCR': 2.1, 'HD95_ED': 3.5, 'HD95_ET': 1.8
        }
        
        # 2. Crear estructura falsa de la Fase 2 para que este script tenga algo que leer
        results_dir = "experiment/phase2_factorial"
        run_dir = os.path.join(results_dir, "unet_seed42")
        os.makedirs(run_dir, exist_ok=True)
        
        # Crear un archivo .pth falso
        pth_path = os.path.join(run_dir, "unet_seed42.pth")
        with open(pth_path, "w") as f:
            f.write("mock weights")
            
        # 3. Ejecutar el orquestador
        from src import run_phase3_evaluate
        try:
            run_phase3_evaluate.main()
            print("✅ Phase 3 orchestrator completed successfully.")
            
            # Comprobar que el CSV de ANOVA clínico se generó correctamente
            csv_path = "experiment/phase3_evaluation/anova_clinical_dataset.csv"
            self.assertTrue(os.path.exists(csv_path), "❌ ERROR: ANOVA CSV file was not created!")
            
            # Verificar que las métricas por clase se convirtieron en columnas
            df = pd.read_csv(csv_path)
            self.assertIn('Dice_NCR', df.columns, "❌ ERROR: Class metrics not found in CSV columns!")
            self.assertEqual(len(df), 1, "❌ ERROR: The CSV should have exactly 1 row (from our mock).")
            
        except Exception as e:
            self.fail(f"❌ Phase 3 orchestrator failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()