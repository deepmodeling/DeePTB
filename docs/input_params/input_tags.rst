.. _`common_options`: 

common_options: 
    | type: ``dict``
    | argument path: ``common_options``

    .. _`common_options/basis`: 

    basis: 
        | type: ``dict``
        | argument path: ``common_options/basis``

        The atomic orbitals used to construct the basis. e.p. {'A':['2s','2p','s*'],'B':'[3s','3p']}

    .. _`common_options/overlap`: 

    overlap: 
        | type: ``bool``, optional, default: ``False``
        | argument path: ``common_options/overlap``

        Whether to calculate the overlap matrix. Default: False

    .. _`common_options/device`: 

    device: 
        | type: ``str``, optional, default: ``cpu``
        | argument path: ``common_options/device``

        The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: `cpu`

    .. _`common_options/dtype`: 

    dtype: 
        | type: ``str``, optional, default: ``float32``
        | argument path: ``common_options/dtype``

        The digital number's precison, choose among: 
                            Default: `float32`
                                - `float32`: indicating torch.float32
                                - `float64`: indicating torch.float64
                

    .. _`common_options/seed`: 

    seed: 
        | type: ``int``, optional, default: ``3982377700``
        | argument path: ``common_options/seed``

        The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`


.. _`train_options`: 

train_options: 
    | type: ``dict``, optional
    | argument path: ``train_options``

    Options that defines the training behaviour of DeePTB.

    .. _`train_options/num_epoch`: 

    num_epoch: 
        | type: ``int``
        | argument path: ``train_options/num_epoch``

        Total number of training epochs. It is worth noted, if the model is reloaded with `-r` or `--restart` option, epoch which have been trained will counted from the time that the checkpoint is saved.

    .. _`train_options/batch_size`: 

    batch_size: 
        | type: ``int``, optional, default: ``1``
        | argument path: ``train_options/batch_size``

        The batch size used in training, Default: 1

    .. _`train_options/ref_batch_size`: 

    ref_batch_size: 
        | type: ``int``, optional, default: ``1``
        | argument path: ``train_options/ref_batch_size``

        The batch size used in reference data, Default: 1

    .. _`train_options/val_batch_size`: 

    val_batch_size: 
        | type: ``int``, optional, default: ``1``
        | argument path: ``train_options/val_batch_size``

        The batch size used in validation data, Default: 1

    .. _`train_options/optimizer`: 

    optimizer: 
        | type: ``dict``, optional, default: ``{}``
        | argument path: ``train_options/optimizer``

                The optimizer setting for selecting the gradient optimizer of model training. Optimizer supported includes `Adam`, `SGD` and `LBFGS` 

                For more information about these optmization algorithm, we refer to:

                - `Adam`: [Adam: A Method for Stochastic Optimization.](https://arxiv.org/abs/1412.6980)

                - `SGD`: [Stochastic Gradient Descent.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)

                - `LBFGS`: [On the limited memory BFGS method for large scale optimization.](http://users.iems.northwestern.edu/~nocedal/PDFfiles/limited-memory.pdf) 

    


        Depending on the value of *type*, different sub args are accepted. 

        .. _`train_options/optimizer/type`: 

        type:
            | type: ``str`` (flag key), default: ``Adam``
            | argument path: ``train_options/optimizer/type`` 
            | possible choices: |code:train_options/optimizer[Adam]|_, |code:train_options/optimizer[SGD]|_

            select type of optimizer, support type includes: `Adam`, `SGD` and `LBFGS`. Default: `Adam`

            .. |code:train_options/optimizer[Adam]| replace:: ``Adam``
            .. _`code:train_options/optimizer[Adam]`: `train_options/optimizer[Adam]`_
            .. |code:train_options/optimizer[SGD]| replace:: ``SGD``
            .. _`code:train_options/optimizer[SGD]`: `train_options/optimizer[SGD]`_

        .. |flag:train_options/optimizer/type| replace:: *type*
        .. _`flag:train_options/optimizer/type`: `train_options/optimizer/type`_


        .. _`train_options/optimizer[Adam]`: 

        When |flag:train_options/optimizer/type|_ is set to ``Adam``: 

        .. _`train_options/optimizer[Adam]/lr`: 

        lr: 
            | type: ``float``, optional, default: ``0.001``
            | argument path: ``train_options/optimizer[Adam]/lr``

            learning rate. Default: 1e-3

        .. _`train_options/optimizer[Adam]/betas`: 

        betas: 
            | type: ``list``, optional, default: ``[0.9, 0.999]``
            | argument path: ``train_options/optimizer[Adam]/betas``

            coefficients used for computing running averages of gradient and its square Default: (0.9, 0.999)

        .. _`train_options/optimizer[Adam]/eps`: 

        eps: 
            | type: ``float``, optional, default: ``1e-08``
            | argument path: ``train_options/optimizer[Adam]/eps``

            term added to the denominator to improve numerical stability, Default: 1e-8

        .. _`train_options/optimizer[Adam]/weight_decay`: 

        weight_decay: 
            | type: ``float``, optional, default: ``0``
            | argument path: ``train_options/optimizer[Adam]/weight_decay``

            weight decay (L2 penalty), Default: 0

        .. _`train_options/optimizer[Adam]/amsgrad`: 

        amsgrad: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``train_options/optimizer[Adam]/amsgrad``

            whether to use the AMSGrad variant of this algorithm from the paper On the [Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) ,Default: False


        .. _`train_options/optimizer[SGD]`: 

        When |flag:train_options/optimizer/type|_ is set to ``SGD``: 

        .. _`train_options/optimizer[SGD]/lr`: 

        lr: 
            | type: ``float``, optional, default: ``0.001``
            | argument path: ``train_options/optimizer[SGD]/lr``

            learning rate. Default: 1e-3

        .. _`train_options/optimizer[SGD]/momentum`: 

        momentum: 
            | type: ``float``, optional, default: ``0.0``
            | argument path: ``train_options/optimizer[SGD]/momentum``

            momentum factor Default: 0

        .. _`train_options/optimizer[SGD]/weight_decay`: 

        weight_decay: 
            | type: ``float``, optional, default: ``0.0``
            | argument path: ``train_options/optimizer[SGD]/weight_decay``

            weight decay (L2 penalty), Default: 0

        .. _`train_options/optimizer[SGD]/dampening`: 

        dampening: 
            | type: ``float``, optional, default: ``0.0``
            | argument path: ``train_options/optimizer[SGD]/dampening``

            dampening for momentum, Default: 0

        .. _`train_options/optimizer[SGD]/nesterov`: 

        nesterov: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``train_options/optimizer[SGD]/nesterov``

            enables Nesterov momentum, Default: False

    .. _`train_options/lr_scheduler`: 

    lr_scheduler: 
        | type: ``dict``, optional, default: ``{}``
        | argument path: ``train_options/lr_scheduler``

        The learning rate scheduler tools settings, the lr scheduler is used to scales down the learning rate during the training process. Proper setting can make the training more stable and efficient. The supported lr schedular includes: `Exponential Decaying (exp)`, `Linear multiplication (linear)`


        Depending on the value of *type*, different sub args are accepted. 

        .. _`train_options/lr_scheduler/type`: 

        type:
            | type: ``str`` (flag key), default: ``exp``
            | argument path: ``train_options/lr_scheduler/type`` 
            | possible choices: |code:train_options/lr_scheduler[exp]|_, |code:train_options/lr_scheduler[linear]|_, |code:train_options/lr_scheduler[rop]|_

            select type of lr_scheduler, support type includes `exp`, `linear`

            .. |code:train_options/lr_scheduler[exp]| replace:: ``exp``
            .. _`code:train_options/lr_scheduler[exp]`: `train_options/lr_scheduler[exp]`_
            .. |code:train_options/lr_scheduler[linear]| replace:: ``linear``
            .. _`code:train_options/lr_scheduler[linear]`: `train_options/lr_scheduler[linear]`_
            .. |code:train_options/lr_scheduler[rop]| replace:: ``rop``
            .. _`code:train_options/lr_scheduler[rop]`: `train_options/lr_scheduler[rop]`_

        .. |flag:train_options/lr_scheduler/type| replace:: *type*
        .. _`flag:train_options/lr_scheduler/type`: `train_options/lr_scheduler/type`_


        .. _`train_options/lr_scheduler[exp]`: 

        When |flag:train_options/lr_scheduler/type|_ is set to ``exp``: 

        .. _`train_options/lr_scheduler[exp]/gamma`: 

        gamma: 
            | type: ``float``, optional, default: ``0.999``
            | argument path: ``train_options/lr_scheduler[exp]/gamma``

            Multiplicative factor of learning rate decay.


        .. _`train_options/lr_scheduler[linear]`: 

        When |flag:train_options/lr_scheduler/type|_ is set to ``linear``: 

        .. _`train_options/lr_scheduler[linear]/start_factor`: 

        start_factor: 
            | type: ``float``, optional, default: ``0.3333333``
            | argument path: ``train_options/lr_scheduler[linear]/start_factor``

            The number we multiply learning rate in the first epoch.         The multiplication factor changes towards end_factor in the following epochs. Default: 1./3.

        .. _`train_options/lr_scheduler[linear]/end_factor`: 

        end_factor: 
            | type: ``float``, optional, default: ``0.3333333``
            | argument path: ``train_options/lr_scheduler[linear]/end_factor``

            The number we multiply learning rate in the first epoch.     The multiplication factor changes towards end_factor in the following epochs. Default: 1./3.

        .. _`train_options/lr_scheduler[linear]/total_iters`: 

        total_iters: 
            | type: ``int``, optional, default: ``5``
            | argument path: ``train_options/lr_scheduler[linear]/total_iters``

            The number of iterations that multiplicative factor reaches to 1. Default: 5.


        .. _`train_options/lr_scheduler[rop]`: 

        When |flag:train_options/lr_scheduler/type|_ is set to ``rop``: 

        rop: reduce on plateau

        .. _`train_options/lr_scheduler[rop]/mode`: 

        mode: 
            | type: ``str``, optional, default: ``min``
            | argument path: ``train_options/lr_scheduler[rop]/mode``

            One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing;         in max mode it will be reduced when the quantity monitored has stopped increasing. Default: 'min'.

        .. _`train_options/lr_scheduler[rop]/factor`: 

        factor: 
            | type: ``float``, optional, default: ``0.1``
            | argument path: ``train_options/lr_scheduler[rop]/factor``

            Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.

        .. _`train_options/lr_scheduler[rop]/patience`: 

        patience: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``train_options/lr_scheduler[rop]/patience``

            Number of epochs with no improvement after which learning rate will be reduced. For example,         if patience = 2, then we will ignore the first 2 epochs with no improvement,         and will only decrease the LR after the 3rd epoch if the loss still hasn't improved then. Default: 10.

        .. _`train_options/lr_scheduler[rop]/threshold`: 

        threshold: 
            | type: ``float``, optional, default: ``0.0001``
            | argument path: ``train_options/lr_scheduler[rop]/threshold``

            Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.

        .. _`train_options/lr_scheduler[rop]/threshold_mode`: 

        threshold_mode: 
            | type: ``str``, optional, default: ``rel``
            | argument path: ``train_options/lr_scheduler[rop]/threshold_mode``

            One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or         best * ( 1 - threshold ) in min mode. In abs mode,         dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: 'rel'.

        .. _`train_options/lr_scheduler[rop]/cooldown`: 

        cooldown: 
            | type: ``int``, optional, default: ``0``
            | argument path: ``train_options/lr_scheduler[rop]/cooldown``

            Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.

        .. _`train_options/lr_scheduler[rop]/min_lr`: 

        min_lr: 
            | type: ``list`` | ``float``, optional, default: ``0``
            | argument path: ``train_options/lr_scheduler[rop]/min_lr``

            A scalar or a list of scalars.         A lower bound on the learning rate of all param groups or each group respectively. Default: 0.

        .. _`train_options/lr_scheduler[rop]/eps`: 

        eps: 
            | type: ``float``, optional, default: ``1e-08``
            | argument path: ``train_options/lr_scheduler[rop]/eps``

            Minimal decay applied to lr.         If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.

    .. _`train_options/save_freq`: 

    save_freq: 
        | type: ``int``, optional, default: ``10``
        | argument path: ``train_options/save_freq``

        Frequency, or every how many iteration to saved the current model into checkpoints, The name of checkpoint is formulated as `latest|best_dptb|nnsk_b<bond_cutoff>_c<sk_cutoff>_w<sk_decay_w>`. Default: `10`

    .. _`train_options/validation_freq`: 

    validation_freq: 
        | type: ``int``, optional, default: ``10``
        | argument path: ``train_options/validation_freq``

        Frequency or every how many iteration to do model validation on validation datasets. Default: `10`

    .. _`train_options/display_freq`: 

    display_freq: 
        | type: ``int``, optional, default: ``1``
        | argument path: ``train_options/display_freq``

        Frequency, or every how many iteration to display the training log to screem. Default: `1`

    .. _`train_options/max_ckpt`: 

    max_ckpt: 
        | type: ``int``, optional, default: ``4``
        | argument path: ``train_options/max_ckpt``

        The maximum number of saved checkpoints, Default: 4

    .. _`train_options/loss_options`: 

    loss_options: 
        | type: ``dict``
        | argument path: ``train_options/loss_options``

        .. _`train_options/loss_options/train`: 

        train: 
            | type: ``dict``
            | argument path: ``train_options/loss_options/train``

            Loss options for training.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`train_options/loss_options/train/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``train_options/loss_options/train/method`` 
                | possible choices: |code:train_options/loss_options/train[hamil]|_, |code:train_options/loss_options/train[eigvals]|_, |code:train_options/loss_options/train[hamil_abs]|_, |code:train_options/loss_options/train[hamil_blas]|_

                The loss function type, defined by a string like `<fitting target>_<loss type>`, Default: `eigs_l2dsf`. supported loss functions includes:

                                    - `eigvals`: The mse loss predicted and labeled eigenvalues and Delta eigenvalues between different k.
                                    - `hamil`: 
                                    - `hamil_abs`:
                                    - `hamil_blas`:
                

                .. |code:train_options/loss_options/train[hamil]| replace:: ``hamil``
                .. _`code:train_options/loss_options/train[hamil]`: `train_options/loss_options/train[hamil]`_
                .. |code:train_options/loss_options/train[eigvals]| replace:: ``eigvals``
                .. _`code:train_options/loss_options/train[eigvals]`: `train_options/loss_options/train[eigvals]`_
                .. |code:train_options/loss_options/train[hamil_abs]| replace:: ``hamil_abs``
                .. _`code:train_options/loss_options/train[hamil_abs]`: `train_options/loss_options/train[hamil_abs]`_
                .. |code:train_options/loss_options/train[hamil_blas]| replace:: ``hamil_blas``
                .. _`code:train_options/loss_options/train[hamil_blas]`: `train_options/loss_options/train[hamil_blas]`_

            .. |flag:train_options/loss_options/train/method| replace:: *method*
            .. _`flag:train_options/loss_options/train/method`: `train_options/loss_options/train/method`_


            .. _`train_options/loss_options/train[hamil]`: 

            When |flag:train_options/loss_options/train/method|_ is set to ``hamil``: 

            .. _`train_options/loss_options/train[hamil]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/train[hamil]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


            .. _`train_options/loss_options/train[eigvals]`: 

            When |flag:train_options/loss_options/train/method|_ is set to ``eigvals``: 

            .. _`train_options/loss_options/train[eigvals]/diff_on`: 

            diff_on: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/train[eigvals]/diff_on``

                Whether to use random differences in loss function. Default: False

            .. _`train_options/loss_options/train[eigvals]/eout_weight`: 

            eout_weight: 
                | type: ``float``, optional, default: ``0.01``
                | argument path: ``train_options/loss_options/train[eigvals]/eout_weight``

                The weight of eigenvalue out of range. Default: 0.01

            .. _`train_options/loss_options/train[eigvals]/diff_weight`: 

            diff_weight: 
                | type: ``float``, optional, default: ``0.01``
                | argument path: ``train_options/loss_options/train[eigvals]/diff_weight``

                The weight of eigenvalue difference. Default: 0.01


            .. _`train_options/loss_options/train[hamil_abs]`: 

            When |flag:train_options/loss_options/train/method|_ is set to ``hamil_abs``: 

            .. _`train_options/loss_options/train[hamil_abs]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/train[hamil_abs]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


            .. _`train_options/loss_options/train[hamil_blas]`: 

            When |flag:train_options/loss_options/train/method|_ is set to ``hamil_blas``: 

            .. _`train_options/loss_options/train[hamil_blas]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/train[hamil_blas]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False

        .. _`train_options/loss_options/validation`: 

        validation: 
            | type: ``dict``, optional
            | argument path: ``train_options/loss_options/validation``

            Loss options for validation.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`train_options/loss_options/validation/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``train_options/loss_options/validation/method`` 
                | possible choices: |code:train_options/loss_options/validation[hamil]|_, |code:train_options/loss_options/validation[eigvals]|_, |code:train_options/loss_options/validation[hamil_abs]|_, |code:train_options/loss_options/validation[hamil_blas]|_

                The loss function type, defined by a string like `<fitting target>_<loss type>`, Default: `eigs_l2dsf`. supported loss functions includes:

                                    - `eigvals`: The mse loss predicted and labeled eigenvalues and Delta eigenvalues between different k.
                                    - `hamil`: 
                                    - `hamil_abs`:
                                    - `hamil_blas`:
                

                .. |code:train_options/loss_options/validation[hamil]| replace:: ``hamil``
                .. _`code:train_options/loss_options/validation[hamil]`: `train_options/loss_options/validation[hamil]`_
                .. |code:train_options/loss_options/validation[eigvals]| replace:: ``eigvals``
                .. _`code:train_options/loss_options/validation[eigvals]`: `train_options/loss_options/validation[eigvals]`_
                .. |code:train_options/loss_options/validation[hamil_abs]| replace:: ``hamil_abs``
                .. _`code:train_options/loss_options/validation[hamil_abs]`: `train_options/loss_options/validation[hamil_abs]`_
                .. |code:train_options/loss_options/validation[hamil_blas]| replace:: ``hamil_blas``
                .. _`code:train_options/loss_options/validation[hamil_blas]`: `train_options/loss_options/validation[hamil_blas]`_

            .. |flag:train_options/loss_options/validation/method| replace:: *method*
            .. _`flag:train_options/loss_options/validation/method`: `train_options/loss_options/validation/method`_


            .. _`train_options/loss_options/validation[hamil]`: 

            When |flag:train_options/loss_options/validation/method|_ is set to ``hamil``: 

            .. _`train_options/loss_options/validation[hamil]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/validation[hamil]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


            .. _`train_options/loss_options/validation[eigvals]`: 

            When |flag:train_options/loss_options/validation/method|_ is set to ``eigvals``: 

            .. _`train_options/loss_options/validation[eigvals]/diff_on`: 

            diff_on: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/validation[eigvals]/diff_on``

                Whether to use random differences in loss function. Default: False

            .. _`train_options/loss_options/validation[eigvals]/eout_weight`: 

            eout_weight: 
                | type: ``float``, optional, default: ``0.01``
                | argument path: ``train_options/loss_options/validation[eigvals]/eout_weight``

                The weight of eigenvalue out of range. Default: 0.01

            .. _`train_options/loss_options/validation[eigvals]/diff_weight`: 

            diff_weight: 
                | type: ``float``, optional, default: ``0.01``
                | argument path: ``train_options/loss_options/validation[eigvals]/diff_weight``

                The weight of eigenvalue difference. Default: 0.01


            .. _`train_options/loss_options/validation[hamil_abs]`: 

            When |flag:train_options/loss_options/validation/method|_ is set to ``hamil_abs``: 

            .. _`train_options/loss_options/validation[hamil_abs]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/validation[hamil_abs]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


            .. _`train_options/loss_options/validation[hamil_blas]`: 

            When |flag:train_options/loss_options/validation/method|_ is set to ``hamil_blas``: 

            .. _`train_options/loss_options/validation[hamil_blas]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/validation[hamil_blas]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False

        .. _`train_options/loss_options/reference`: 

        reference: 
            | type: ``dict``, optional
            | argument path: ``train_options/loss_options/reference``

            Loss options for reference data in training.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`train_options/loss_options/reference/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``train_options/loss_options/reference/method`` 
                | possible choices: |code:train_options/loss_options/reference[hamil]|_, |code:train_options/loss_options/reference[eigvals]|_, |code:train_options/loss_options/reference[hamil_abs]|_, |code:train_options/loss_options/reference[hamil_blas]|_

                The loss function type, defined by a string like `<fitting target>_<loss type>`, Default: `eigs_l2dsf`. supported loss functions includes:

                                    - `eigvals`: The mse loss predicted and labeled eigenvalues and Delta eigenvalues between different k.
                                    - `hamil`: 
                                    - `hamil_abs`:
                                    - `hamil_blas`:
                

                .. |code:train_options/loss_options/reference[hamil]| replace:: ``hamil``
                .. _`code:train_options/loss_options/reference[hamil]`: `train_options/loss_options/reference[hamil]`_
                .. |code:train_options/loss_options/reference[eigvals]| replace:: ``eigvals``
                .. _`code:train_options/loss_options/reference[eigvals]`: `train_options/loss_options/reference[eigvals]`_
                .. |code:train_options/loss_options/reference[hamil_abs]| replace:: ``hamil_abs``
                .. _`code:train_options/loss_options/reference[hamil_abs]`: `train_options/loss_options/reference[hamil_abs]`_
                .. |code:train_options/loss_options/reference[hamil_blas]| replace:: ``hamil_blas``
                .. _`code:train_options/loss_options/reference[hamil_blas]`: `train_options/loss_options/reference[hamil_blas]`_

            .. |flag:train_options/loss_options/reference/method| replace:: *method*
            .. _`flag:train_options/loss_options/reference/method`: `train_options/loss_options/reference/method`_


            .. _`train_options/loss_options/reference[hamil]`: 

            When |flag:train_options/loss_options/reference/method|_ is set to ``hamil``: 

            .. _`train_options/loss_options/reference[hamil]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/reference[hamil]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


            .. _`train_options/loss_options/reference[eigvals]`: 

            When |flag:train_options/loss_options/reference/method|_ is set to ``eigvals``: 

            .. _`train_options/loss_options/reference[eigvals]/diff_on`: 

            diff_on: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/reference[eigvals]/diff_on``

                Whether to use random differences in loss function. Default: False

            .. _`train_options/loss_options/reference[eigvals]/eout_weight`: 

            eout_weight: 
                | type: ``float``, optional, default: ``0.01``
                | argument path: ``train_options/loss_options/reference[eigvals]/eout_weight``

                The weight of eigenvalue out of range. Default: 0.01

            .. _`train_options/loss_options/reference[eigvals]/diff_weight`: 

            diff_weight: 
                | type: ``float``, optional, default: ``0.01``
                | argument path: ``train_options/loss_options/reference[eigvals]/diff_weight``

                The weight of eigenvalue difference. Default: 0.01


            .. _`train_options/loss_options/reference[hamil_abs]`: 

            When |flag:train_options/loss_options/reference/method|_ is set to ``hamil_abs``: 

            .. _`train_options/loss_options/reference[hamil_abs]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/reference[hamil_abs]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


            .. _`train_options/loss_options/reference[hamil_blas]`: 

            When |flag:train_options/loss_options/reference/method|_ is set to ``hamil_blas``: 

            .. _`train_options/loss_options/reference[hamil_blas]/onsite_shift`: 

            onsite_shift: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``train_options/loss_options/reference[hamil_blas]/onsite_shift``

                Whether to use onsite shift in loss function. Default: False


.. _`data_options`: 

data_options: 
    | type: ``dict``
    | argument path: ``data_options``

    The options for dataset settings in training.

    .. _`data_options/train`: 

    train: 
        | type: ``dict``
        | argument path: ``data_options/train``

        The dataset settings for training.

        .. _`data_options/train/type`: 

        type: 
            | type: ``str``, optional, default: ``DefaultDataset``
            | argument path: ``data_options/train/type``

            The type of dataset.

        .. _`data_options/train/root`: 

        root: 
            | type: ``str``
            | argument path: ``data_options/train/root``

            This is where the dataset stores data files.

        .. _`data_options/train/prefix`: 

        prefix: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``data_options/train/prefix``

            The prefix of the folders under root, which will be loaded in dataset.

        .. _`data_options/train/separator`: 

        separator: 
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/train/separator``

            the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'

        .. _`data_options/train/get_Hamiltonian`: 

        get_Hamiltonian: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_Hamiltonian``

            Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset.

        .. _`data_options/train/get_overlap`: 

        get_overlap: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_overlap``

            Choose whether the overlap blocks are loaded when building dataset.

        .. _`data_options/train/get_DM`: 

        get_DM: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_DM``

            Choose whether the density matrix is loaded when building dataset.

        .. _`data_options/train/get_eigenvalues`: 

        get_eigenvalues: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/train/get_eigenvalues``

            Choose whether the eigenvalues and k-points are loaded when building dataset.

    .. _`data_options/validation`: 

    validation: 
        | type: ``dict``, optional
        | argument path: ``data_options/validation``

        The dataset settings for validation.

        .. _`data_options/validation/type`: 

        type: 
            | type: ``str``, optional, default: ``DefaultDataset``
            | argument path: ``data_options/validation/type``

            The type of dataset.

        .. _`data_options/validation/root`: 

        root: 
            | type: ``str``
            | argument path: ``data_options/validation/root``

            This is where the dataset stores data files.

        .. _`data_options/validation/prefix`: 

        prefix: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``data_options/validation/prefix``

            The prefix of the folders under root, which will be loaded in dataset.

        .. _`data_options/validation/separator`: 

        separator: 
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/validation/separator``

            the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'

        .. _`data_options/validation/get_Hamiltonian`: 

        get_Hamiltonian: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_Hamiltonian``

            Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset.

        .. _`data_options/validation/get_overlap`: 

        get_overlap: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_overlap``

            Choose whether the overlap blocks are loaded when building dataset.

        .. _`data_options/validation/get_DM`: 

        get_DM: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_DM``

            Choose whether the density matrix is loaded when building dataset.

        .. _`data_options/validation/get_eigenvalues`: 

        get_eigenvalues: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/validation/get_eigenvalues``

            Choose whether the eigenvalues and k-points are loaded when building dataset.

    .. _`data_options/reference`: 

    reference: 
        | type: ``dict``, optional
        | argument path: ``data_options/reference``

        The dataset settings for reference.

        .. _`data_options/reference/type`: 

        type: 
            | type: ``str``, optional, default: ``DefaultDataset``
            | argument path: ``data_options/reference/type``

            The type of dataset.

        .. _`data_options/reference/root`: 

        root: 
            | type: ``str``
            | argument path: ``data_options/reference/root``

            This is where the dataset stores data files.

        .. _`data_options/reference/prefix`: 

        prefix: 
            | type: ``str`` | ``NoneType``, optional, default: ``None``
            | argument path: ``data_options/reference/prefix``

            The prefix of the folders under root, which will be loaded in dataset.

        .. _`data_options/reference/separator`: 

        separator: 
            | type: ``str``, optional, default: ``.``
            | argument path: ``data_options/reference/separator``

            the sepatator used to separate the prefix and suffix in the dataset directory. Default: '.'

        .. _`data_options/reference/get_Hamiltonian`: 

        get_Hamiltonian: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_Hamiltonian``

            Choose whether the Hamiltonian blocks (and overlap blocks, if provided) are loaded when building dataset.

        .. _`data_options/reference/get_overlap`: 

        get_overlap: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_overlap``

            Choose whether the overlap blocks are loaded when building dataset.

        .. _`data_options/reference/get_DM`: 

        get_DM: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_DM``

            Choose whether the density matrix is loaded when building dataset.

        .. _`data_options/reference/get_eigenvalues`: 

        get_eigenvalues: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``data_options/reference/get_eigenvalues``

            Choose whether the eigenvalues and k-points are loaded when building dataset.


.. _`model_options`: 

model_options: 
    | type: ``dict``, optional
    | argument path: ``model_options``

    The parameters to define the `nnsk`,`mix` and `dptb` model.

    .. _`model_options/embedding`: 

    embedding: 
        | type: ``dict``, optional
        | argument path: ``model_options/embedding``

        The parameters to define the embedding model.


        Depending on the value of *method*, different sub args are accepted. 

        .. _`model_options/embedding/method`: 

        method:
            | type: ``str`` (flag key), default: ``se2``
            | argument path: ``model_options/embedding/method`` 
            | possible choices: |code:model_options/embedding[se2]|_, |code:model_options/embedding[baseline]|_, |code:model_options/embedding[deeph-e3]|_, |code:model_options/embedding[e3baseline_0]|_, |code:model_options/embedding[e3baseline_1]|_, |code:model_options/embedding[e3baseline_2]|_, |code:model_options/embedding[e3baseline_3]|_, |code:model_options/embedding[e3baseline_4]|_, |code:model_options/embedding[e3baseline_5]|_, |code:model_options/embedding[e3baseline_6]|_, |code:model_options/embedding[e3baseline_nonlocal]|_

            The parameters to define the embedding model.

            .. |code:model_options/embedding[se2]| replace:: ``se2``
            .. _`code:model_options/embedding[se2]`: `model_options/embedding[se2]`_
            .. |code:model_options/embedding[baseline]| replace:: ``baseline``
            .. _`code:model_options/embedding[baseline]`: `model_options/embedding[baseline]`_
            .. |code:model_options/embedding[deeph-e3]| replace:: ``deeph-e3``
            .. _`code:model_options/embedding[deeph-e3]`: `model_options/embedding[deeph-e3]`_
            .. |code:model_options/embedding[e3baseline_0]| replace:: ``e3baseline_0``
            .. _`code:model_options/embedding[e3baseline_0]`: `model_options/embedding[e3baseline_0]`_
            .. |code:model_options/embedding[e3baseline_1]| replace:: ``e3baseline_1``
            .. _`code:model_options/embedding[e3baseline_1]`: `model_options/embedding[e3baseline_1]`_
            .. |code:model_options/embedding[e3baseline_2]| replace:: ``e3baseline_2``
            .. _`code:model_options/embedding[e3baseline_2]`: `model_options/embedding[e3baseline_2]`_
            .. |code:model_options/embedding[e3baseline_3]| replace:: ``e3baseline_3``
            .. _`code:model_options/embedding[e3baseline_3]`: `model_options/embedding[e3baseline_3]`_
            .. |code:model_options/embedding[e3baseline_4]| replace:: ``e3baseline_4``
            .. _`code:model_options/embedding[e3baseline_4]`: `model_options/embedding[e3baseline_4]`_
            .. |code:model_options/embedding[e3baseline_5]| replace:: ``e3baseline_5``
            .. _`code:model_options/embedding[e3baseline_5]`: `model_options/embedding[e3baseline_5]`_
            .. |code:model_options/embedding[e3baseline_6]| replace:: ``e3baseline_6``
            .. _`code:model_options/embedding[e3baseline_6]`: `model_options/embedding[e3baseline_6]`_
            .. |code:model_options/embedding[e3baseline_nonlocal]| replace:: ``e3baseline_nonlocal``
            .. _`code:model_options/embedding[e3baseline_nonlocal]`: `model_options/embedding[e3baseline_nonlocal]`_

        .. |flag:model_options/embedding/method| replace:: *method*
        .. _`flag:model_options/embedding/method`: `model_options/embedding/method`_


        .. _`model_options/embedding[se2]`: 

        When |flag:model_options/embedding/method|_ is set to ``se2``: 

        .. _`model_options/embedding[se2]/rs`: 

        rs: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[se2]/rs``

            The soft cutoff where the smooth function starts.

        .. _`model_options/embedding[se2]/rc`: 

        rc: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[se2]/rc``

            The hard cutoff where the smooth function value ~0.0

        .. _`model_options/embedding[se2]/radial_net`: 

        radial_net: 
            | type: ``dict``
            | argument path: ``model_options/embedding[se2]/radial_net``

            network to build the descriptors.

            .. _`model_options/embedding[se2]/radial_net/neurons`: 

            neurons: 
                | type: ``list``
                | argument path: ``model_options/embedding[se2]/radial_net/neurons``

                the size of nn for descriptor

            .. _`model_options/embedding[se2]/radial_net/activation`: 

            activation: 
                | type: ``str``, optional, default: ``tanh``
                | argument path: ``model_options/embedding[se2]/radial_net/activation``

                activation

            .. _`model_options/embedding[se2]/radial_net/if_batch_normalized`: 

            if_batch_normalized: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``model_options/embedding[se2]/radial_net/if_batch_normalized``

                whether to turn on the batch normalization.

        .. _`model_options/embedding[se2]/n_axis`: 

        n_axis: 
            | type: ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[se2]/n_axis``

            the out axis shape of the deepmd-se2 descriptor.


        .. _`model_options/embedding[baseline]`: 

        When |flag:model_options/embedding/method|_ is set to ``baseline``: 

        .. _`model_options/embedding[baseline]/p`: 

        p: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[baseline]/p``

        .. _`model_options/embedding[baseline]/rc`: 

        rc: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[baseline]/rc``

        .. _`model_options/embedding[baseline]/n_basis`: 

        n_basis: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_basis``

        .. _`model_options/embedding[baseline]/n_radial`: 

        n_radial: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_radial``

        .. _`model_options/embedding[baseline]/n_sqrt_radial`: 

        n_sqrt_radial: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_sqrt_radial``

        .. _`model_options/embedding[baseline]/n_layer`: 

        n_layer: 
            | type: ``int``
            | argument path: ``model_options/embedding[baseline]/n_layer``

        .. _`model_options/embedding[baseline]/radial_net`: 

        radial_net: 
            | type: ``dict``
            | argument path: ``model_options/embedding[baseline]/radial_net``

            .. _`model_options/embedding[baseline]/radial_net/neurons`: 

            neurons: 
                | type: ``list``
                | argument path: ``model_options/embedding[baseline]/radial_net/neurons``

            .. _`model_options/embedding[baseline]/radial_net/activation`: 

            activation: 
                | type: ``str``, optional, default: ``tanh``
                | argument path: ``model_options/embedding[baseline]/radial_net/activation``

            .. _`model_options/embedding[baseline]/radial_net/if_batch_normalized`: 

            if_batch_normalized: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``model_options/embedding[baseline]/radial_net/if_batch_normalized``

        .. _`model_options/embedding[baseline]/hidden_net`: 

        hidden_net: 
            | type: ``dict``
            | argument path: ``model_options/embedding[baseline]/hidden_net``

            .. _`model_options/embedding[baseline]/hidden_net/neurons`: 

            neurons: 
                | type: ``list``
                | argument path: ``model_options/embedding[baseline]/hidden_net/neurons``

            .. _`model_options/embedding[baseline]/hidden_net/activation`: 

            activation: 
                | type: ``str``, optional, default: ``tanh``
                | argument path: ``model_options/embedding[baseline]/hidden_net/activation``

            .. _`model_options/embedding[baseline]/hidden_net/if_batch_normalized`: 

            if_batch_normalized: 
                | type: ``bool``, optional, default: ``False``
                | argument path: ``model_options/embedding[baseline]/hidden_net/if_batch_normalized``

        .. _`model_options/embedding[baseline]/n_axis`: 

        n_axis: 
            | type: ``int`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[baseline]/n_axis``


        .. _`model_options/embedding[deeph-e3]`: 

        When |flag:model_options/embedding/method|_ is set to ``deeph-e3``: 

        .. _`model_options/embedding[deeph-e3]/irreps_embed`: 

        irreps_embed: 
            | type: ``str``, optional, default: ``64x0e``
            | argument path: ``model_options/embedding[deeph-e3]/irreps_embed``

        .. _`model_options/embedding[deeph-e3]/irreps_mid`: 

        irreps_mid: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[deeph-e3]/irreps_mid``

        .. _`model_options/embedding[deeph-e3]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[deeph-e3]/lmax``

        .. _`model_options/embedding[deeph-e3]/n_basis`: 

        n_basis: 
            | type: ``int``, optional, default: ``128``
            | argument path: ``model_options/embedding[deeph-e3]/n_basis``

        .. _`model_options/embedding[deeph-e3]/rc`: 

        rc: 
            | type: ``float``
            | argument path: ``model_options/embedding[deeph-e3]/rc``

        .. _`model_options/embedding[deeph-e3]/n_layer`: 

        n_layer: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[deeph-e3]/n_layer``


        .. _`model_options/embedding[e3baseline_0]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_0``: 

        .. _`model_options/embedding[e3baseline_0]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_0]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_0]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_0]/lmax``

        .. _`model_options/embedding[e3baseline_0]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_0]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_0]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_0]/r_max``

        .. _`model_options/embedding[e3baseline_0]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_0]/n_layers``

        .. _`model_options/embedding[e3baseline_0]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_0]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_0]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_0]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_0]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_0]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_0]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_0]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_0]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_0]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_0]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_0]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_1]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_1``: 

        .. _`model_options/embedding[e3baseline_1]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_1]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_1]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_1]/lmax``

        .. _`model_options/embedding[e3baseline_1]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_1]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_1]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_1]/r_max``

        .. _`model_options/embedding[e3baseline_1]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_1]/n_layers``

        .. _`model_options/embedding[e3baseline_1]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_1]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_1]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_1]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_1]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_1]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_1]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_1]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_1]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_1]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_1]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_1]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_2]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_2``: 

        .. _`model_options/embedding[e3baseline_2]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_2]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_2]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_2]/lmax``

        .. _`model_options/embedding[e3baseline_2]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_2]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_2]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_2]/r_max``

        .. _`model_options/embedding[e3baseline_2]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_2]/n_layers``

        .. _`model_options/embedding[e3baseline_2]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_2]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_2]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_2]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_2]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_2]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_2]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_2]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_2]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_2]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_2]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_2]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_3]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_3``: 

        .. _`model_options/embedding[e3baseline_3]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_3]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_3]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_3]/lmax``

        .. _`model_options/embedding[e3baseline_3]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_3]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_3]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_3]/r_max``

        .. _`model_options/embedding[e3baseline_3]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_3]/n_layers``

        .. _`model_options/embedding[e3baseline_3]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_3]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_3]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_3]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_3]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_3]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_3]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_3]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_3]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_3]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_3]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_3]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_4]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_4``: 

        .. _`model_options/embedding[e3baseline_4]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``, optional, default: ``64x0e+32x1o+16x2e+8x3o+8x4e+4x5o``
            | argument path: ``model_options/embedding[e3baseline_4]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_4]/lmax`: 

        lmax: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_4]/lmax``

        .. _`model_options/embedding[e3baseline_4]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``, optional, default: ``50``
            | argument path: ``model_options/embedding[e3baseline_4]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_4]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_4]/r_max``

        .. _`model_options/embedding[e3baseline_4]/n_layers`: 

        n_layers: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_4]/n_layers``

        .. _`model_options/embedding[e3baseline_4]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``3``
            | argument path: ``model_options/embedding[e3baseline_4]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_4]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_4]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_4]/latent_kwargs`: 

        latent_kwargs: 
            | type: ``dict`` | ``NoneType``, optional, default: ``None``
            | argument path: ``model_options/embedding[e3baseline_4]/latent_kwargs``

        .. _`model_options/embedding[e3baseline_4]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_4]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_4]/linear_after_env_embed`: 

        linear_after_env_embed: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_4]/linear_after_env_embed``

        .. _`model_options/embedding[e3baseline_4]/latent_resnet_update_ratios_learnable`: 

        latent_resnet_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_4]/latent_resnet_update_ratios_learnable``


        .. _`model_options/embedding[e3baseline_5]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_5``: 

        .. _`model_options/embedding[e3baseline_5]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``
            | argument path: ``model_options/embedding[e3baseline_5]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_5]/lmax`: 

        lmax: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/lmax``

        .. _`model_options/embedding[e3baseline_5]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_5]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/r_max``

        .. _`model_options/embedding[e3baseline_5]/n_layers`: 

        n_layers: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_5]/n_layers``

        .. _`model_options/embedding[e3baseline_5]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``model_options/embedding[e3baseline_5]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_5]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_5]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_5]/cutoff_type`: 

        cutoff_type: 
            | type: ``str``, optional, default: ``polynomial``
            | argument path: ``model_options/embedding[e3baseline_5]/cutoff_type``

            The type of cutoff function. Default: polynomial

        .. _`model_options/embedding[e3baseline_5]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_5]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_5]/tp_radial_emb`: 

        tp_radial_emb: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_5]/tp_radial_emb``

            Whether to use tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_5]/tp_radial_channels`: 

        tp_radial_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_5]/tp_radial_channels``

            The number of channels in tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_5]/latent_channels`: 

        latent_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_5]/latent_channels``

            The number of channels in latent embedding.

        .. _`model_options/embedding[e3baseline_5]/latent_dim`: 

        latent_dim: 
            | type: ``int``, optional, default: ``256``
            | argument path: ``model_options/embedding[e3baseline_5]/latent_dim``

            The dimension of latent embedding.

        .. _`model_options/embedding[e3baseline_5]/res_update`: 

        res_update: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model_options/embedding[e3baseline_5]/res_update``

            Whether to use residual update.

        .. _`model_options/embedding[e3baseline_5]/res_update_ratios`: 

        res_update_ratios: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model_options/embedding[e3baseline_5]/res_update_ratios``

            The ratios of residual update, should in (0,1).

        .. _`model_options/embedding[e3baseline_5]/res_update_ratios_learnable`: 

        res_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_5]/res_update_ratios_learnable``

            Whether to make the ratios of residual update learnable.


        .. _`model_options/embedding[e3baseline_6]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_6``: 

        .. _`model_options/embedding[e3baseline_6]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``
            | argument path: ``model_options/embedding[e3baseline_6]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_6]/lmax`: 

        lmax: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/lmax``

        .. _`model_options/embedding[e3baseline_6]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_6]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/r_max``

        .. _`model_options/embedding[e3baseline_6]/n_layers`: 

        n_layers: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_6]/n_layers``

        .. _`model_options/embedding[e3baseline_6]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``model_options/embedding[e3baseline_6]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_6]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_6]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_6]/cutoff_type`: 

        cutoff_type: 
            | type: ``str``, optional, default: ``polynomial``
            | argument path: ``model_options/embedding[e3baseline_6]/cutoff_type``

            The type of cutoff function. Default: polynomial

        .. _`model_options/embedding[e3baseline_6]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_6]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_6]/tp_radial_emb`: 

        tp_radial_emb: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_6]/tp_radial_emb``

            Whether to use tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_6]/tp_radial_channels`: 

        tp_radial_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_6]/tp_radial_channels``

            The number of channels in tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_6]/latent_channels`: 

        latent_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_6]/latent_channels``

            The number of channels in latent embedding.

        .. _`model_options/embedding[e3baseline_6]/latent_dim`: 

        latent_dim: 
            | type: ``int``, optional, default: ``256``
            | argument path: ``model_options/embedding[e3baseline_6]/latent_dim``

            The dimension of latent embedding.

        .. _`model_options/embedding[e3baseline_6]/res_update`: 

        res_update: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model_options/embedding[e3baseline_6]/res_update``

            Whether to use residual update.

        .. _`model_options/embedding[e3baseline_6]/res_update_ratios`: 

        res_update_ratios: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model_options/embedding[e3baseline_6]/res_update_ratios``

            The ratios of residual update, should in (0,1).

        .. _`model_options/embedding[e3baseline_6]/res_update_ratios_learnable`: 

        res_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_6]/res_update_ratios_learnable``

            Whether to make the ratios of residual update learnable.


        .. _`model_options/embedding[e3baseline_nonlocal]`: 

        When |flag:model_options/embedding/method|_ is set to ``e3baseline_nonlocal``: 

        .. _`model_options/embedding[e3baseline_nonlocal]/irreps_hidden`: 

        irreps_hidden: 
            | type: ``str``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/irreps_hidden``

        .. _`model_options/embedding[e3baseline_nonlocal]/lmax`: 

        lmax: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/lmax``

        .. _`model_options/embedding[e3baseline_nonlocal]/avg_num_neighbors`: 

        avg_num_neighbors: 
            | type: ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/avg_num_neighbors``

        .. _`model_options/embedding[e3baseline_nonlocal]/r_max`: 

        r_max: 
            | type: ``dict`` | ``float`` | ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/r_max``

        .. _`model_options/embedding[e3baseline_nonlocal]/n_layers`: 

        n_layers: 
            | type: ``int``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/n_layers``

        .. _`model_options/embedding[e3baseline_nonlocal]/n_radial_basis`: 

        n_radial_basis: 
            | type: ``int``, optional, default: ``10``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/n_radial_basis``

        .. _`model_options/embedding[e3baseline_nonlocal]/PolynomialCutoff_p`: 

        PolynomialCutoff_p: 
            | type: ``int``, optional, default: ``6``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/PolynomialCutoff_p``

            The order of polynomial cutoff function. Default: 6

        .. _`model_options/embedding[e3baseline_nonlocal]/cutoff_type`: 

        cutoff_type: 
            | type: ``str``, optional, default: ``polynomial``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/cutoff_type``

            The type of cutoff function. Default: polynomial

        .. _`model_options/embedding[e3baseline_nonlocal]/env_embed_multiplicity`: 

        env_embed_multiplicity: 
            | type: ``int``, optional, default: ``1``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/env_embed_multiplicity``

        .. _`model_options/embedding[e3baseline_nonlocal]/tp_radial_emb`: 

        tp_radial_emb: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/tp_radial_emb``

            Whether to use tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/tp_radial_channels`: 

        tp_radial_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/tp_radial_channels``

            The number of channels in tensor product radial embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/latent_channels`: 

        latent_channels: 
            | type: ``list``, optional, default: ``[128, 128]``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/latent_channels``

            The number of channels in latent embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/latent_dim`: 

        latent_dim: 
            | type: ``int``, optional, default: ``256``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/latent_dim``

            The dimension of latent embedding.

        .. _`model_options/embedding[e3baseline_nonlocal]/res_update`: 

        res_update: 
            | type: ``bool``, optional, default: ``True``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/res_update``

            Whether to use residual update.

        .. _`model_options/embedding[e3baseline_nonlocal]/res_update_ratios`: 

        res_update_ratios: 
            | type: ``float``, optional, default: ``0.5``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/res_update_ratios``

            The ratios of residual update, should in (0,1).

        .. _`model_options/embedding[e3baseline_nonlocal]/res_update_ratios_learnable`: 

        res_update_ratios_learnable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/embedding[e3baseline_nonlocal]/res_update_ratios_learnable``

            Whether to make the ratios of residual update learnable.

    .. _`model_options/prediction`: 

    prediction: 
        | type: ``dict``, optional
        | argument path: ``model_options/prediction``

        The parameters to define the prediction model


        Depending on the value of *method*, different sub args are accepted. 

        .. _`model_options/prediction/method`: 

        method:
            | type: ``str`` (flag key)
            | argument path: ``model_options/prediction/method`` 
            | possible choices: |code:model_options/prediction[sktb]|_, |code:model_options/prediction[e3tb]|_

            The options to indicate the prediction model. Can be sktb or e3tb.

            .. |code:model_options/prediction[sktb]| replace:: ``sktb``
            .. _`code:model_options/prediction[sktb]`: `model_options/prediction[sktb]`_
            .. |code:model_options/prediction[e3tb]| replace:: ``e3tb``
            .. _`code:model_options/prediction[e3tb]`: `model_options/prediction[e3tb]`_

        .. |flag:model_options/prediction/method| replace:: *method*
        .. _`flag:model_options/prediction/method`: `model_options/prediction/method`_


        .. _`model_options/prediction[sktb]`: 

        When |flag:model_options/prediction/method|_ is set to ``sktb``: 

        neural network options for prediction model.

        .. _`model_options/prediction[sktb]/neurons`: 

        neurons: 
            | type: ``list``
            | argument path: ``model_options/prediction[sktb]/neurons``

            neurons in the neural network.

        .. _`model_options/prediction[sktb]/activation`: 

        activation: 
            | type: ``str``, optional, default: ``tanh``
            | argument path: ``model_options/prediction[sktb]/activation``

            activation function.

        .. _`model_options/prediction[sktb]/if_batch_normalized`: 

        if_batch_normalized: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/prediction[sktb]/if_batch_normalized``

            if to turn on batch normalization


        .. _`model_options/prediction[e3tb]`: 

        When |flag:model_options/prediction/method|_ is set to ``e3tb``: 

        neural network options for prediction model.

        .. _`model_options/prediction[e3tb]/scales_trainable`: 

        scales_trainable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/prediction[e3tb]/scales_trainable``

            whether to scale the trianing target.

        .. _`model_options/prediction[e3tb]/shifts_trainable`: 

        shifts_trainable: 
            | type: ``bool``, optional, default: ``False``
            | argument path: ``model_options/prediction[e3tb]/shifts_trainable``

            whether to shift the training target.

    .. _`model_options/nnsk`: 

    nnsk: 
        | type: ``dict``, optional
        | argument path: ``model_options/nnsk``

        The parameters to define the nnsk model.

        .. _`model_options/nnsk/onsite`: 

        onsite: 
            | type: ``dict``
            | argument path: ``model_options/nnsk/onsite``

            The onsite options to define the onsite of nnsk model.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`model_options/nnsk/onsite/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``model_options/nnsk/onsite/method`` 
                | possible choices: |code:model_options/nnsk/onsite[strain]|_, |code:model_options/nnsk/onsite[uniform]|_, |code:model_options/nnsk/onsite[NRL]|_, |code:model_options/nnsk/onsite[none]|_

                The onsite correction mode, the onsite energy is expressed as the energy of isolated atoms plus the model correction, the correction mode are:
                                    Default: `none`: use the database onsite energy value.
                                    - `strain`: The strain mode correct the onsite matrix densly by $$H_{i,i}^{lm,l^\prime m^\prime} = \epsilon_l^0 \delta_{ll^\prime}\delta_{mm^\prime} + \sum_p \sum_{\zeta} \Big[ \mathcal{U}_{\zeta}(\hat{r}_{ip}) \ \epsilon_{ll^\prime \zeta} \Big]_{mm^\prime}$$ which is also parameterized as a set of Slater-Koster like integrals.

                                    - `uniform`: The correction is a energy shift respect of orbital of each atom. Which is formally written as: 
                                                $$H_{i,i}^{lm,l^\prime m^\prime} = (\epsilon_l^0+\epsilon_l^\prime) \delta_{ll^\prime}\delta_{mm^\prime}$$ Where $\epsilon_l^0$ is the isolated energy level from the DeePTB onsite database, and $\epsilon_l^\prime$ is the parameters to fit.
                                    - `NRL`: use the NRL-TB formula.
                

                .. |code:model_options/nnsk/onsite[strain]| replace:: ``strain``
                .. _`code:model_options/nnsk/onsite[strain]`: `model_options/nnsk/onsite[strain]`_
                .. |code:model_options/nnsk/onsite[uniform]| replace:: ``uniform``
                .. _`code:model_options/nnsk/onsite[uniform]`: `model_options/nnsk/onsite[uniform]`_
                .. |code:model_options/nnsk/onsite[NRL]| replace:: ``NRL``
                .. _`code:model_options/nnsk/onsite[NRL]`: `model_options/nnsk/onsite[NRL]`_
                .. |code:model_options/nnsk/onsite[none]| replace:: ``none``
                .. _`code:model_options/nnsk/onsite[none]`: `model_options/nnsk/onsite[none]`_

            .. |flag:model_options/nnsk/onsite/method| replace:: *method*
            .. _`flag:model_options/nnsk/onsite/method`: `model_options/nnsk/onsite/method`_


            .. _`model_options/nnsk/onsite[strain]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``strain``: 

            .. _`model_options/nnsk/onsite[strain]/rs`: 

            rs: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/onsite[strain]/rs``

                The smooth cutoff `fc` for strain model. rs is where fc = 0.5

            .. _`model_options/nnsk/onsite[strain]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/onsite[strain]/w``

                The decay factor of `fc` for strain and nrl model.


            .. _`model_options/nnsk/onsite[uniform]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``uniform``: 



            .. _`model_options/nnsk/onsite[NRL]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``NRL``: 

            .. _`model_options/nnsk/onsite[NRL]/rc`: 

            rc: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/onsite[NRL]/rc``

                The smooth cutoff of `fc` for nrl model, rc is where fc ~ 0.0

            .. _`model_options/nnsk/onsite[NRL]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/onsite[NRL]/w``

                The decay factor of `fc` for strain and nrl model.

            .. _`model_options/nnsk/onsite[NRL]/lda`: 

            lda: 
                | type: ``float``, optional, default: ``1.0``
                | argument path: ``model_options/nnsk/onsite[NRL]/lda``

                The lambda type encoding value in nrl model. now only support elementary substance


            .. _`model_options/nnsk/onsite[none]`: 

            When |flag:model_options/nnsk/onsite/method|_ is set to ``none``: 


        .. _`model_options/nnsk/hopping`: 

        hopping: 
            | type: ``dict``
            | argument path: ``model_options/nnsk/hopping``

            The hopping options to define the hopping of nnsk model.


            Depending on the value of *method*, different sub args are accepted. 

            .. _`model_options/nnsk/hopping/method`: 

            method:
                | type: ``str`` (flag key)
                | argument path: ``model_options/nnsk/hopping/method`` 
                | possible choices: |code:model_options/nnsk/hopping[powerlaw]|_, |code:model_options/nnsk/hopping[varTang96]|_, |code:model_options/nnsk/hopping[NRL0]|_, |code:model_options/nnsk/hopping[NRL1]|_, |code:model_options/nnsk/hopping[custom]|_

                The hopping formula. 
                                    -  `powerlaw`: the powerlaw formula for bond length dependence for sk integrals.
                                    -  `varTang96`: a variational formula based on Tang96 formula.
                                    -  `NRL0`: the old version of NRL formula for overlap, we set overlap and hopping share same options.
                                    -  `NRL1`: the new version of NRL formula for overlap. 
                    

                .. |code:model_options/nnsk/hopping[powerlaw]| replace:: ``powerlaw``
                .. _`code:model_options/nnsk/hopping[powerlaw]`: `model_options/nnsk/hopping[powerlaw]`_
                .. |code:model_options/nnsk/hopping[varTang96]| replace:: ``varTang96``
                .. _`code:model_options/nnsk/hopping[varTang96]`: `model_options/nnsk/hopping[varTang96]`_
                .. |code:model_options/nnsk/hopping[NRL0]| replace:: ``NRL0``
                .. _`code:model_options/nnsk/hopping[NRL0]`: `model_options/nnsk/hopping[NRL0]`_
                .. |code:model_options/nnsk/hopping[NRL1]| replace:: ``NRL1``
                .. _`code:model_options/nnsk/hopping[NRL1]`: `model_options/nnsk/hopping[NRL1]`_
                .. |code:model_options/nnsk/hopping[custom]| replace:: ``custom``
                .. _`code:model_options/nnsk/hopping[custom]`: `model_options/nnsk/hopping[custom]`_

            .. |flag:model_options/nnsk/hopping/method| replace:: *method*
            .. _`flag:model_options/nnsk/hopping/method`: `model_options/nnsk/hopping/method`_


            .. _`model_options/nnsk/hopping[powerlaw]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``powerlaw``: 

            .. _`model_options/nnsk/hopping[powerlaw]/rs`: 

            rs: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[powerlaw]/rs``

                The cut-off for smooth function fc for powerlaw and varTang96, fc(rs)=0.5

            .. _`model_options/nnsk/hopping[powerlaw]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[powerlaw]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[varTang96]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``varTang96``: 

            .. _`model_options/nnsk/hopping[varTang96]/rs`: 

            rs: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[varTang96]/rs``

                The cut-off for smooth function fc for powerlaw and varTang96, fc(rs)=0.5

            .. _`model_options/nnsk/hopping[varTang96]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[varTang96]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[NRL0]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``NRL0``: 

            .. _`model_options/nnsk/hopping[NRL0]/rc`: 

            rc: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[NRL0]/rc``

                The cut-off for smooth function fc for NRL, fc(rc) = 0.

            .. _`model_options/nnsk/hopping[NRL0]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[NRL0]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[NRL1]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``NRL1``: 

            .. _`model_options/nnsk/hopping[NRL1]/rc`: 

            rc: 
                | type: ``float``, optional, default: ``6.0``
                | argument path: ``model_options/nnsk/hopping[NRL1]/rc``

                The cut-off for smooth function fc for NRL, fc(rc) = 0.

            .. _`model_options/nnsk/hopping[NRL1]/w`: 

            w: 
                | type: ``float``, optional, default: ``0.1``
                | argument path: ``model_options/nnsk/hopping[NRL1]/w``

                 The decay w in fc


            .. _`model_options/nnsk/hopping[custom]`: 

            When |flag:model_options/nnsk/hopping/method|_ is set to ``custom``: 


        .. _`model_options/nnsk/soc`: 

        soc: 
            | type: ``dict``, optional, default: ``{}``
            | argument path: ``model_options/nnsk/soc``

            The soc options to define the soc of nnsk model,
                            Default: {} # empty dict

                            - {'method':'none'} : use database soc value. 
                            - {'method':uniform} : set lambda_il; assign a soc lambda value for each orbital -l on each atomtype i; l=0,1,2 for s p d.

        .. _`model_options/nnsk/freeze`: 

        freeze: 
            | type: ``str`` | ``bool`` | ``list``, optional, default: ``False``
            | argument path: ``model_options/nnsk/freeze``

            The parameters to define the freeze of nnsk model can be bool and string and list.

                                Default: False

                                 - True: freeze all the nnsk parameters

                                 - False: train all the nnsk parameters
 
                                 - 'hopping','onsite','overlap' and 'soc' to freeze the corresponding parameters.
                                 - list of the strings e.g. ['overlap','soc'] to freeze both overlap and soc parameters.

        .. _`model_options/nnsk/std`: 

        std: 
            | type: ``float``, optional, default: ``0.01``
            | argument path: ``model_options/nnsk/std``

            The std value to initialize the nnsk parameters. Default: 0.01

        .. _`model_options/nnsk/push`: 

        push: 
            | type: ``dict`` | ``bool``, optional, default: ``False``
            | argument path: ``model_options/nnsk/push``

            The parameters to define the push the soft cutoff of nnsk model.

            .. _`model_options/nnsk/push/rs_thr`: 

            rs_thr: 
                | type: ``float`` | ``int``, optional, default: ``0.0``
                | argument path: ``model_options/nnsk/push/rs_thr``

                The step size for cutoff value for smooth function in the nnsk anlytical formula.

            .. _`model_options/nnsk/push/rc_thr`: 

            rc_thr: 
                | type: ``float`` | ``int``, optional, default: ``0.0``
                | argument path: ``model_options/nnsk/push/rc_thr``

                The step size for cutoff value for smooth function in the nnsk anlytical formula.

            .. _`model_options/nnsk/push/w_thr`: 

            w_thr: 
                | type: ``float`` | ``int``, optional, default: ``0.0``
                | argument path: ``model_options/nnsk/push/w_thr``

                The step size for decay factor w.

            .. _`model_options/nnsk/push/period`: 

            period: 
                | type: ``int``, optional, default: ``100``
                | argument path: ``model_options/nnsk/push/period``

                the interval of iterations to modify the rs w values.

