<div align="center">

# Enhancing Sample Utilization in Noise-Robust Deep Metric Learning With Subgroup-Based Positive-Pair Selection

<p align="center">
<a href="https://arxiv.org/abs/2312.09147"><img src="https://img.shields.io/badge/Arxiv-2312.09147-B31B1B.svg"></a>
</p>


</div>

---
Official implementation of IEEE Transactions on Image Processing paper **[Enhancing Sample Utilization in Noise-Robust Deep Metric Learning With Subgroup-Based Positive-Pair Selection](https://arxiv.org/abs/2312.09147)**.


## ⭐️ Key Features
- An asynchronous server that stores all sample features, enabling it to continuously select positive training samples for each training session in real-time.
- A noise-robust deep metric learning framework to enhance information retrieval in computer vision tasks.

### Preparing Dataset
Please refer to **[PRISM](https://github.com/alibaba-edu/Ranking-based-Instance-Selection)** to download the CARS_98N dataset.
For other datasets, we highly recommend downloading them from 
**[OpenDataLab](https://opendatalab.com/)**.

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --cfg configs/sample_config.yaml 
```

## Acknowledgements
- We would like to thank [@Jiaheng Liu](https://liujiaheng.github.io/) for helpful discussion.
- Our PCS module is modified from [PRISM](https://github.com/alibaba-edu/Ranking-based-Instance-Selection) and [HFSoftmax](https://github.com/yl-1993/hfsoftmax)

## Citation

If you find this work helpful, please consider citing our paper:
```bibtex
@article{yu2024enhancing,
  title={Enhancing Sample Utilization in Noise-Robust Deep Metric Learning With Subgroup-Based Positive-Pair Selection},
  author={Yu, Zhipeng and Xu, Qianqian and Jiang, Yangbangyan and Sun, Yingfei and Huang, Qingming},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  pages={6083--6097},
  year={2024},
  publisher={IEEE}
}
```