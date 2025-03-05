# DTP-Attack
A decision-based trajectory prediction black-box adversarial attack.

<img src="/image/method_DTPAttack.png" width="50%" height="50%">

Comparison of the principles of different attack methods.

<img src="/image/method_compare.png" width="400" height="200">

## Requirements

* Python 3.8+

Install necessary packages.
```
pip install -r requirements.txt
```
## Prepare datasets
place datasets in directory `/data` following `README.md` in `data/apolloscape`, `data/NGSIM`, and `data/nuScenes`.
translate raw dataset into JSON-format testing data.

## Prepare models
The models should be placed in `/data/${model_name}_${dataset_name}/model/${attack_mode}`.

## Run normal prediction as well as the SA-Attack
```
python DTP-Attack.py --help
```
# Results of different number of queries
<img src="/image/different_query.png" width="50%" height="50%">
