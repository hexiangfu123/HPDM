# HPDM

## Setup

### Requirements
```Python==3.11```

```Pytorch==2.0.1```

```transformers==4.34.1```

```scikit-learn==1.3.2```

```tqdm==4.65.0```

```fire==0.5.0```

### Dataset
We validate the effectiveness of our model in the real world dataset [MIND](https://msnews.github.io/) and [Adressa](https://dl.acm.org/doi/pdf/10.1145/3106426.3109436)

|Datasets|#news|#user|#categories|#impressions|#click
|-|-|-|-|-|-|
|MIND-large|161,013|1,000,000|20|15,777,377|24,155,470
|Adressa|48486|3083438|24|-|27223576
_
## Training

process the download dataset MIND
```
python data_pro.py large bert
```



training model with multi GPU (cuda:0,1,2,3)
```
sh run1.sh run HPDM cumsum-sum nce 1,2,3,4 large HPDM bert
```

## Reference


He X, Peng Q, Liu H, et al. HPDM: A Hierarchical Popularity-aware Debiased Modeling Approach for Personalized News Recommender [C]. In Proceedings of the 34th International Joint Conference on Artificial Intelligence, 2025: 1-9

### bibtex
```
@inproceedings{Wu2019naml,
  author = {He, Xiangfu and Peng, Qiyao and Liu, hongtao and Shao, Minglai},
  booktitle = {Proceedings of the 34th International Joint Conference on Artificial Intelligence},
  pages = {1-9},
  title = {HPDM: A Hierarchical Popularity-aware Debiased Modeling Approach for Personalized News Recommender},
  year = {2025}
}
```