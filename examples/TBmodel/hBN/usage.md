1.  Prepare data 
    ```shell
    $ cd data
    $ python3 ./prepare_dat.py -in ./set.0
    ```
2. Train TB model.
    ```shell
    $ cd train
    $ dptb train  
    ```
    for from scratch case, while for restart.
    ```shell
    $ dptb train  -r  restart
    ```
3. Check TB with DFT band structure.
    ```shell
    $ cp train/checkpoint.pl  check
    $ cd check
    $ python3 check_band.py
    ```
    check_band.py will ask for some input paras. just press Enter to use default.