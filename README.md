# ADL22-HW3
Dataset & evaluation script for ADL 2022 homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Download trained models
```shell
# Please make sure you have change the argument of the downloaded model for later testing
bash download.sh
```
The downloaded model would be a directory and it locates at `./final.ckpt`.

## Training
```shell
python train.py
```
or

```shell
python trainRL.py
```

## Inference
```shell
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```