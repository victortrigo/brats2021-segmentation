import os
import sys
import yaml
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath('src'))

class TestPhase2Factorial(unittest.TestCase):
    """Test suite for Phase 2 (Factorial Experiment Orchestrator)."""
    
    @patch('run_phase2_factorial.DataLoader')
    @patch('run_phase2_factorial.Dataset')
    @patch('run_phase2_factorial.run_experiment')
    def test_fase2_orchestrator(self, mock_run_experiment, mock_dataset, mock_dataloader):
        print("\n" + "="*50)
        print("🧪 STARTING TEST: PHASE 2 (Orchestrator Logic)")
        print("="*50)
        
        mock_dataset.return_value = MagicMock()
        mock_dataloader.return_value = [] 
        mock_run_experiment.return_value = (0.85, 0.75)
        
        hpo_dir = "experiment/phase1_hpo"
        os.makedirs(hpo_dir, exist_ok=True)
        test_yaml_path = f"{hpo_dir}/best_params_unet.yaml"
        with open(test_yaml_path, "w") as f:
            yaml.dump({"batch_size": 2, "lr": 0.001, "optimizer": "Adam"}, f)
            
        with patch('run_phase2_factorial.SEEDS', [42]):
            import run_phase2_factorial
            try:
                run_phase2_factorial.main() # Correrá UNet y saltará elegantemente los demás
                print("✅ Phase 2 orchestrator completed successfully.")
                
                csv_path = "experiment/phase2_factorial/factorial_results.csv"
                self.assertTrue(os.path.exists(csv_path), "❌ ERROR: CSV file was not created!")
                
            except Exception as e:
                self.fail(f"❌ Phase 2 orchestrator failed with exception: {e}")
            # with patch('src.run_phase2_factorial.modelos', ["UNet"]):
            #     try:
            #         run_phase2_factorial.main()
            #         print("✅ Phase 2 orchestrator completed successfully.")
            #         csv_path = "experiment/phase2_factorial/factorial_results.csv"
            #         self.assertTrue(os.path.exists(csv_path), "❌ ERROR: CSV file was not created!")
            #     except Exception as e:
            #         self.fail(f"❌ Phase 2 orchestrator failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()