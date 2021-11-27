# DIRECT


## Data simulation
```sh
$ python data_synthetic.py
```

## ITE estimation
```sh
$ python main_disent.py --dataset synthetic --K 4 --dim_zt 32 --dim_zi 32 --lr 1e-3 --beta 20 --epochs 300 
```

[1] Jing Ma, Ruocheng Guo, Aidong Zhang, Jundong Li, “Multi-Cause Effect Estimation with Disentangled Confounder Representation”, International Joint Conference on Artificial Intelligence (IJCAI), 2021.
