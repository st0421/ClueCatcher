#### Train
using [train.py](train.py), being careful this time to specify the architecture's name **with** the ST suffix and to insert as *--init* argument the path to the weights of the feature extractor trained at the previous step. You will end up running something like `python train.py --net ClueCatcher --traindb (choices=split.available_datasets) --valdb (choices=split.available_datasets) --face scale --size 224 --batch 32 --lr 1e-5 --valint 500 --patience 10 --maxiter 10000 --seed 41 --models_dir weights/cluecatcher`

### Test 
using [test.py](test.py). You will end up running something like `python test_model_ours.py --model_path weights/cluecatcher --testsets (choices=split.available_datasets) --ffpp_faces_df_path /your/ff++/faces/dataframe/path --ffpp_faces_dir /your/ff++/faces/directory --testsplits --device 0`

### Result
You can find Jupyter notebooks for results computations in the [notebook](notebook) folder.

<p align='center'>
  <img src="demo/scoreid0_0002.gif" width="300" height='180'/>
  <img src="demo/scroeid0_id3_0002.gif" width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/scroeid0_id20_0002.gif"  width="300" height='180'/>
  <img src="demo/scroeid0_id23_0002.gif"  width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/scroeid38_0004.gif" width="300" height='180'/>
  <img src="demo/scroeid38_id23_0004.gif"  width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/scroeid38_id26_0004.gif" width="300" height='180'/>
  <img src="demo/scroeid38_id28_0004.gif" width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/281.gif" width="300" height='180'/>
  <img src="demo/281_df.gif"  width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/281_f2f.gif" width="300" height='180'/>
  <img src="demo/281_nt.gif" width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/997.gif" width="300" height='180'/>
  <img src="demo/997_df.gif"  width="300" height='180'/>
</p>
<p align='center'>
  <img src="demo/997_fs.gif" width="300" height='180'/>
  <img src="demo/997_f2f.gif" width="300" height='180'/>
</p>
