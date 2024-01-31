> 本代码使用开源框架OpenSTL训练、测试
>
> `@article{tan2023openstl,`
>   `title={OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning},`
>   `author={Tan, Cheng and Li, Siyuan and Gao, Zhangyang and Guan, Wenfei and Wang, Zedong and Liu, Zicheng and Wu, Lirong and Li, Stan Z},`
>   `journal={arXiv preprint arXiv:2306.11249},`
>   `year={2023}`
> `}`





## get started

1.将`OpenSTL/openstl/methods/simvp.py`中的代码替换为`MSTANet_loss.py`中的代码

2.将`OpenSTL/openstl/models/simvp_model.py`中的代码替换为`MSTANet_model.py`中的代码

3.train/test

`bash tools/prepare_data/download_mmnist.sh`

`python tools/train.py -d mmnist --lr 2e-3 -c configs/mmnist/simvp/SimVP_VAN.py --ex_name mmnist_mstanet`

`python tools/test.py -d mmnist -c configs/mmnist/simvp/SimVP_VAN.py --ex_name mmnist_mstanet`

