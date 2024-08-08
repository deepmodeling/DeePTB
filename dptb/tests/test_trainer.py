import pytest
from dptb.nnops.trainer import Trainer
import os
from pathlib import Path
from dptb.utils.argcheck import normalize,collect_cutoffs
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
    cutoffops = collect_cutoffs(jdata)
    train_datasets = build_dataset(**cutoffops, **jdata["data_options"]["train"], **jdata["common_options"])


    
    def for_init(self,trainer):
        assert isinstance(trainer, Trainer)
    
    def for_iteration(self,trainer, batch, ref_batch=None):
        loss = trainer.iteration(batch, ref_batch)
        assert loss.ndim == 0
        assert isinstance(loss.item(), float)
    
    def for_epoch(self,trainer, expect_ref=False):
        assert trainer.use_reference == expect_ref
        try : trainer.epoch()
        except : raise AssertionError("The epoch method failed to execute")


    def test_fromscratch_noref_noval(self):
        run_options = self.run_options
        jdata = self.jdata
        train_datasets = self.train_datasets
        model = build_model(None, model_options=jdata["model_options"], 
                        common_options=jdata["common_options"])
        trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=None,
            reference_datasets=None)

        self.for_init(trainer)
        batch = next(iter(trainer.train_loader))
        self.for_iteration(trainer, batch)
        self.for_epoch(trainer, expect_ref=False)

    
    def test_fromscratch_ref_noval(self):
        run_options = self.run_options
        jdata = self.jdata
        jdata["data_options"]["reference"] = jdata["data_options"]["train"]
        jdata["train_options"]["ref_batch_size"] = jdata["train_options"]["batch_size"]
        jdata["train_options"]["loss_options"]["reference"] = jdata["train_options"]["loss_options"]["train"]
        train_datasets = self.train_datasets

        reference_datasets = build_dataset(**self.cutoffops,**jdata["data_options"]["reference"], **jdata["common_options"])

        model = build_model(None, model_options=jdata["model_options"], 
                        common_options=jdata["common_options"])
        
        trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=None,
            reference_datasets=reference_datasets)
        
        self.for_init(trainer)
        batch = next(iter(trainer.train_loader))
        ref_batch = next(iter(trainer.reference_loader))
        self.for_iteration(trainer, batch, ref_batch)
        self.for_epoch(trainer, expect_ref=True)
    
    def test_fromscratch_noref_val(self):
        run_options = self.run_options
        jdata = self.jdata
        jdata["data_options"]["validation"] = jdata["data_options"]["train"]
        jdata["train_options"]["val_batch_size"] = jdata["train_options"]["batch_size"]
        jdata["train_options"]["loss_options"]["validation"] = jdata["train_options"]["loss_options"]["train"]
        train_datasets = self.train_datasets

        validation_datasets = build_dataset(**self.cutoffops,**jdata["data_options"]["validation"], **jdata["common_options"])

        model = build_model(None, model_options=jdata["model_options"], 
                        common_options=jdata["common_options"])
        
        trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=validation_datasets,
            reference_datasets=None)
                
        self.for_init(trainer)
        batch = next(iter(trainer.train_loader))
        self.for_iteration(trainer, batch, ref_batch=None)
        self.for_epoch(trainer, expect_ref=False)

        loss = trainer.validation(fast=True)
        assert loss.ndim == 0
        assert isinstance(loss.item(), float)

    def test_initmodel_noref_nval(self):
        run_options = self.run_options
        jdata = self.jdata
        train_datasets = self.train_datasets
        checkpoint = f"{rootdir}/test_sktb/output/test_valence/checkpoint/nnsk.best.pth"
        run_options.update({"init_model": checkpoint, "restart": None})
        model = build_model(checkpoint, model_options=jdata["model_options"], 
                            common_options=jdata["common_options"])
        trainer = Trainer(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            model = model,
            train_datasets=train_datasets,
            validation_datasets=None,
            reference_datasets=None)
        
        self.for_init(trainer)
        batch = next(iter(trainer.train_loader))
        self.for_iteration(trainer, batch, ref_batch=None)
        self.for_epoch(trainer, expect_ref=False)

    # def test_initmodel_fail(self):
    #     run_options = self.run_options
    #     jdata = self.jdata
    #     train_datasets = self.train_datasets``
    #     checkpoint = f"{rootdir}/test_sktb/output/test_valence/checkpoint/nnsk.best.pth"
    #     run_options.update({"init_model": checkpoint, "restart": checkpoint})
    #     with pytest.raises(AssertionError):
    #         model = build_model(checkpoint, model_options=jdata["model_options"], 
    #                         common_options=jdata["common_options"], statistics=train_datasets.E3statistics())
    
    def test_restart_noref_nval(self):
        run_options = self.run_options
        jdata = self.jdata
        train_datasets = self.train_datasets
        checkpoint = f"{rootdir}/test_sktb/output/test_valence/checkpoint/nnsk.best.pth"
        run_options.update({"init_model": None, "restart": checkpoint})
        trainer = Trainer.restart(
            train_options=jdata["train_options"],
            common_options=jdata["common_options"],
            checkpoint=checkpoint,
            train_datasets=train_datasets,
            reference_datasets=None,
            validation_datasets=None,
        )
        self.for_init(trainer)
        batch = next(iter(trainer.train_loader))
        self.for_iteration(trainer, batch, ref_batch=None)
        self.for_epoch(trainer, expect_ref=False)
