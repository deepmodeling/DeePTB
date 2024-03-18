import pytest
from dptb.nnops.trainer import Trainer
import os
from pathlib import Path
from dptb.utils.argcheck import normalize
from dptb.utils.tools import j_loader
from dptb.nn.build import build_model
from dptb.data.build import build_dataset

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

class TestTrainer:
        
    run_options = {
        "init_model": None,
        "restart": None,
        "train_soc": False,
        "log_path": f"{rootdir}/out/log.txt",
        "log_level": "INFO"
    }
        
    INPUT_file = f"{rootdir}/test_sktb/input/input_valence.json"
    output = f"{rootdir}/out"
    
    jdata = j_loader(INPUT_file)
    jdata = normalize(jdata)
    train_datasets = build_dataset(set_options=jdata["data_options"]["train"], common_options=jdata["common_options"])

    model = build_model(run_options=run_options, model_options=jdata["model_options"], common_options=jdata["common_options"], statistics=train_datasets.E3statistics())

    trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=None,
            reference_datasets=None)
    
    def test_init(self):
        assert isinstance(self.trainer, Trainer)
    
    def test_iteration(self):
        batch = next(iter(self.trainer.train_loader))
        ref_batch = None
        loss = self.trainer.iteration(batch, ref_batch)
        assert loss.ndim == 0
        assert isinstance(loss.item(), float)
#def test_iteration(trainer):
#    # Test the iteration method of the Trainer class
#    batch = None
#    ref_batch = None
#    loss = trainer.iteration(batch, ref_batch)
#    assert isinstance(loss, float)
#
#def test_restart(trainer):
#    # Test the restart method of the Trainer class
#    checkpoint = "path/to/checkpoint.pth"
#    train_datasets = None
#    train_options = {}
#    common_options = {}
#    reference_datasets = None
#    validation_datasets = None
#
#    restarted_trainer = Trainer.restart(checkpoint, train_datasets, train_options, common_options, reference_datasets, validation_datasets)
#    assert isinstance(restarted_trainer, Trainer)
#
#def test_epoch(trainer):
#    # Test the epoch method of the Trainer class
#    trainer.epoch()
#    # Add assertions here to check the expected behavior of the epoch method
#
#def test_validation(trainer):
#    # Test the validation method of the Trainer class
#    loss = trainer.validation()
#    assert isinstance(loss, float)import unittest
#from unittest.mock import MagicMock
#from dptb.nnops.trainer import Trainer
#
#class TestTrainer(unittest.TestCase):
#
#    def setUp(self):
#        # Create mock objects for the dependencies
#        self.model = MagicMock()
#        self.train_datasets = MagicMock()
#        self.reference_datasets = MagicMock()
#        self.validation_datasets = MagicMock()
#        self.train_options = {}
#        self.common_options = {}
#
#    def test_iteration(self):
#        # Create an instance of the Trainer class
#        trainer = Trainer(
#            model=self.model,
#            train_datasets=self.train_datasets,
#            reference_datasets=self.reference_datasets,
#            validation_datasets=self.validation_datasets,
#            train_options=self.train_options,
#            common_options=self.common_options
#        )
#
#        # Call the iteration method and assert the expected behavior
#        batch = MagicMock()
#        ref_batch = MagicMock()
#        loss = trainer.iteration(batch, ref_batch)
#        self.assertIsNotNone(loss)
#
#    def test_restart(self):
#        # Create an instance of the Trainer class
#        trainer = Trainer(
#            model=self.model,
#            train_datasets=self.train_datasets,
#            reference_datasets=self.reference_datasets,
#            validation_datasets=self.validation_datasets,
#            train_options=self.train_options,
#            common_options=self.common_options
#        )
#
#        # Call the restart method and assert the expected behavior
#        checkpoint = "path/to/checkpoint"
#        restarted_trainer = trainer.restart(
#            checkpoint=checkpoint,
#            train_datasets=self.train_datasets,
#            train_options=self.train_options,
#            common_options=self.common_options,
#            reference_datasets=self.reference_datasets,
#            validation_datasets=self.validation_datasets
#        )
#        self.assertIsNotNone(restarted_trainer)
#
#    def test_epoch(self):
#        # Create an instance of the Trainer class
#        trainer = Trainer(
#            model=self.model,
#            train_datasets=self.train_datasets,
#            reference_datasets=self.reference_datasets,
#            validation_datasets=self.validation_datasets,
#            train_options=self.train_options,
#            common_options=self.common_options
#        )
#
#        # Call the epoch method and assert the expected behavior
#        trainer.epoch()
#
#    def test_validation(self):
#        # Create an instance of the Trainer class
#        trainer = Trainer(
#            model=self.model,
#            train_datasets=self.train_datasets,
#            reference_datasets=self.reference_datasets,
#            validation_datasets=self.validation_datasets,
#            train_options=self.train_options,
#            common_options=self.common_options
#        )
#
#        # Call the validation method and assert the expected behavior
#        loss = trainer.validation(fast=True)
#        self.assertIsNotNone(loss)
#
#if __name__ == '__main__':
#    unittest.main()