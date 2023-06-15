### Data Preprocessing method
[Video Face Manipulation Detection Through Ensemble of CNNs](https://github.com/polimi-ispl/icpr2020dfdc/tree/master)

#### Train
using [train.py](train.py), being careful this time to specify the architecture's name **with** the ST suffix and to insert as *--init* argument the path to the weights of the feature extractor trained at the previous step. You will end up running something like `python train.py --net ClueCatcher --traindb (choices=split.available_datasets) --valdb (choices=split.available_datasets) --face scale --size 224 --batch 32 --lr 1e-5 --valint 500 --patience 10 --maxiter 10000 --seed 41 --models_dir weights/cluecatcher`

### Test 
using [test.py](test.py). You will end up running something like `python test_model_ours.py --model_path weights/cluecatcher --testsets (choices=split.available_datasets) --ffpp_faces_df_path /your/ff++/faces/dataframe/path --ffpp_faces_dir /your/ff++/faces/directory --testsplits --device 0`

### Result
You can find Jupyter notebooks for results computations in the [notebook](notebook) folder.

![id0_0002](https://github.com/st0421/ClueCatcher/assets/81230496/37b3baa3-6393-4081-8392-09b30fcc66dc)
![id0_id23_0002](https://github.com/st0421/ClueCatcher/assets/81230496/52223e05-8eed-467a-811d-1d9e9089fe70)
![id0_id20_0002](https://github.com/st0421/ClueCatcher/assets/81230496/ffae01d8-c723-463b-9906-b7f43ce636fb)
![id0_id3_0002](https://github.com/st0421/ClueCatcher/assets/81230496/ec3b6eed-8288-4405-9a3f-7733e8b71726)
