> This code uses the open-source framework OpenSTL for training and testing
>
> `@article{tan2023openstl,`
>   `title={OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning},`
>   `author={Tan, Cheng and Li, Siyuan and Gao, Zhangyang and Guan, Wenfei and Wang, Zedong and Liu, Zicheng and Wu, Lirong and Li, Stan Z},`
>   `journal={arXiv preprint arXiv:2306.11249},`
>   `year={2023}`
> `}`





## get started

1.Replace the code in`OpenSTL/openstl/methods/simvp.py`with the code in`MSTANet_loss.py`

2.Replace the code in`OpenSTL/openstl/models/simvp_model.py`with the code in`MSTANet_model.py`

3.train/test

`bash tools/prepare_data/download_mmnist.sh`

`python tools/train.py -d mmnist --lr 2e-3 -c configs/mmnist/simvp/SimVP_VAN.py --ex_name mmnist_mstanet`

`python tools/test.py -d mmnist -c configs/mmnist/simvp/SimVP_VAN.py --ex_name mmnist_mstanet`

