========================================
Train Options
========================================
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

