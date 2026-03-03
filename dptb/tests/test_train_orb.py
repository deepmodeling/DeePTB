import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import sys
from dptb.entrypoints.train import train
train_module = sys.modules['dptb.entrypoints.train']

class TestTrainOrbitalIntegration(unittest.TestCase):
    
    @patch.object(train_module, 'j_loader')
    @patch.object(train_module, 'normalize')
    @patch.object(train_module, 'build_dataset')
    @patch.object(train_module, 'build_model')
    @patch.object(train_module, 'Trainer')
    @patch.object(train_module, 'set_log_handles')
    @patch.object(train_module, 'collect_cutoffs')
    @patch.object(train_module, 'setup_seed')
    @patch('os.makedirs') # Mock makedirs to prevent creating directories
    @patch('pathlib.Path.mkdir')
    def test_train_orbital_parsing(self, mock_mkdir, mock_makedirs, mock_setup_seed, mock_collect_cutoffs, mock_set_log, mock_trainer, mock_build_model, mock_build_dataset, mock_normalize, mock_j_loader):
        # Use the specific orbital file requested
        orb_file_path = "./dptb/tests/data/e3_band/data/Si_gga_7au_100Ry_2s2p1d.orb"
        
        # Verify file exists before running test
        if not os.path.exists(orb_file_path):
             self.skipTest(f"Test file not found: {orb_file_path}")

        try:
            # Mock j_loader to return our config
            mock_config = {
                "common_options": {
                    "basis": {"Si": orb_file_path},
                    "dtype": "float32",
                    "seed": 123
                },
                "model_options": {
                    "prediction": {"method": "e3tb"}
                },
                "data_options": {"train": {}},
                "train_options": {}
            }
            mock_j_loader.return_value = mock_config
            mock_normalize.return_value = mock_config # Assume normalize returns it as is for this test

            # Run train
            # passing output=None prevents file creation attempts
            try:
                # We expect it to eventually crash or finish. 
                # Since we mocked build_model etc, it might proceed until it tries to use the mocked objects.
                # However, the orbital parsing happens early.
                # raising an exception in build_model allows us to stop execution after parsing
                mock_build_model.side_effect = InterruptedError("Verify point reached")
                
                train(INPUT="dummy.json", init_model=None, restart=None, output=None, log_level=20, log_path=None)
            except InterruptedError:
                pass
            except Exception as e:
                self.fail(f"Train raised unexpected exception: {e}")

            # Verify jdata was modified
            # Since jdata is a local variable in train, we can't inspect it directly.
            # But we can inspect the calls to build_model or build_dataset, which receive jdata components.
            
            # Check build_model call args
            args, kwargs = mock_build_model.call_args
            common_options = kwargs.get('common_options', {})
            
            # 1. Check basis string
            self.assertEqual(common_options['basis']['Si'], '2s2p1d')
            
            # 2. Check orbital_files_content
            self.assertIn('orbital_files_content', common_options)
            self.assertIn('Si', common_options['orbital_files_content'])
            # Verify some content from the real file
            self.assertIn("Number of Sorbital-->       2", common_options['orbital_files_content']['Si'])

        finally:
            pass 
    
    @patch.object(train_module, 'j_loader')
    @patch.object(train_module, 'normalize')
    def test_train_orbital_parsing_wrong_method(self, mock_normalize, mock_j_loader):
         with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.orb') as tmp_orb:
            tmp_orb_path = tmp_orb.name
         try:
            mock_config = {
                "common_options": {
                    "basis": {"Si": tmp_orb_path},
                    "dtype": "float32"
                },
                "model_options": {
                    "prediction": {"method": "sktb"} # Not e3tb
                }
            }
            mock_j_loader.return_value = mock_config
            mock_normalize.return_value = mock_config

            with self.assertRaisesRegex(ValueError, "only supported for the 'e3tb' method"):
                train(INPUT="dummy.json", init_model=None, restart=None, output=None, log_level=20, log_path=None)

         finally:
            if os.path.exists(tmp_orb_path):
                os.remove(tmp_orb_path)

if __name__ == '__main__':
    unittest.main()
