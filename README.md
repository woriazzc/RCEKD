# Understanding and Unleashing the Potential of Cross-Entropy Loss in Knowledge Distillation for Recommender Systems

This repo provide the Pytorch for TSCKD.

### Requirements

Python 3.9

```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### Experiments

1. First, you need to train the teacher. For example,

   ```shell
   python -u main.py --dataset=citeulike --S_backbone=bpr --train_teacher --suffix teacher
   ```

   You can replace "citeulike" with "gowalla" and "yelp" to test on your interested dataset.

   You can also set "--S_backbone=lightgcn" or "--S_backbone=hstu".

2. Now, you can start knowledge distillation. For example,

   ```shell
   python -u main.py --dataset=citeulike --S_backbone=bpr --T_backbone=bpr --model=tsckd
   ```

   By configuring the "model", you can test other KD methods, such as rrd.

We provide some exemplar command lines in run.sh.

### Notes

In configs/, we have provided the configuration of hyperparameters for TSCKD, together hyperparameters for other compared methods.

In modeling/KD/baseline.py, we provide the codes for all baseline methods.

The code for TSCKD is given in modeling/KD/playground.py.

The codes for all backbones are provided in modeling/backbone/.