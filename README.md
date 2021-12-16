# Extreme Dynamic Classifier Chains
Classifier chains is a key technique in multi-label classification, sinceit allows to consider label dependencies effectively. However, the classifiers arealigned according to a static order of the labels. In the concept of **dynamic classifier chains (DCC)** the label ordering is chosen for each prediction dynamically depending on the respective instance at hand. We combine this concept with the boosting of **extreme gradient boosted trees (XGBoot)**, an effective and scalable state-of-the-art technique, and incorporate DCC in a fast multi-label extension of XGBoost which we make publicly available. As only positive labels have to be predicted and these are usually only few, the training costs can be further substantially reduced. Moreover, as experiments on ten datasets show, the length of the chain allows for a more control over the usage of previous predictions and hence over the measure one want to optimize,

If you want to use the code, or just cite the paper, please use the following:
```
@ARTICLE{loza21DCC,
        author = {Loza Menc{\'{\i}}a, Eneldo and Kulessa, Moritz and Bohlender, Simon and F{\"{u}}rnkranz, Johannes},
         title = {Tree-Based Dynamic Classifier Chains},
       journal = {Machine Learning Journal},
          year = {2021},
           url = {https://arxiv.org/abs/2112.06672},
          note = {To be published}
}
@INPROCEEDINGS{bohlender20XDCC,
       author = {Bohlender, Simon and Loza Menc{\'{\i}}a, Eneldo and Kulessa, Moritz},
        month = oct,
        title = {Extreme Gradient Boosted Multi-label Trees for Dynamic Classifier Chains},
    booktitle = {Discovery Science - 23rd International Conference, {DS} 2020, Thessaloniki, Greece, October 19-21, 2020, Proceedings},
       series = {Lecture Notes in Computer Science},
       volume = {12323},
         year = {2020},
        pages = {471--485},
    publisher = {Springer International Publishing},
          url = {https://arxiv.org/abs/2006.08094},
          doi = {10.1007/978-3-030-61527-7_31},
}
``` 


# Installation

The first step requires to build the modified multilabel version of **XGBoost** and install the resulting python package to build the dynamic chain model. This requires MinGW, i.e. the `mingw32-make` command, and Python 3.
To start the build run the following commands:

    cd XGBoost_ML
    mingw32-make -j4

After a successful execution the python package can be installed. 

    cd python-package
    python setup.py install
You should now be able to import the package into your Python project:

    import xgboost as xgb


# Training the Dynamic Chain Model

We recommend running the models by calling `train_dcc.py` from within a console.
Place all datasets as `.arff` files into the `datasets` directory. Append `-train` to the train set and `-test` to the test set.

## Parameters:
The following parameters are available:
|Parameter   |Short  |Description| Required|
|--|--|--|--|
|`--filename <string>`|`-f` | Name of your dataset .arff file located in the datasets sub-directory |yes|
|`--num_labels <int>`|`-l`|Number of Labels in the dataset|yes|
|`--models <string>`|`-m`|Specifies all models that will be build. Available options: <ul><li>`dcc`: The proposed dynamic chain model</li><li>`sxgb`: A single multilabel XGBoost model</li><li>`cc-dcc`: A classifier chain with the label order of a previously built dynamic chain</li><li>`cc-freq`: A classifier chain with a label order sorted by label frequency (frequent to rare) in the train set</li><li>`cc-rare`: A classifier chain with a label order sorted by label frequency (rare to frequent) in the train set</li><li>`cc-rand`: A classifier chain with a random label order</li><li>`br`: A binary relevance model</li></ul> example: `-m "dc,br"`|yes|
|`--validation <int>`|`-v`|Size of validation set. The first XX% of the train set will be used for validating the model. If the parameter is not set, the test set will be used for evaluation. Example: `--validation 20` The frist 20% will be used for evaluation, the last 80% for training. (default: 0)|no|
|`--max_depth <int>`|`-d`|Max depth of each XGBoost multilabel tree (default: 10)|no|
|`--num_rounds <int>`|`-r`|Number of boosting rounds of each XGBoost model (default: 10)| no|
|`--chain_length <int>`|`-c`| Length of the chain. Represents number of labeling-rounds. Each round builds a new XGBoost model that will predict a single label per instance (default: num_labels)| no|
|`--split <int>`|`-s`|Index of split method used for building the trees. Available options: <ul><li>maxGain: 1</li><li>maxWeight: 2</li><li>sumGain: 3</li><li>sumWeight: 4</li><li>maxAbsGain: 5</li><li>sumAbsGain: 6</li></ul> (default: 1)|no|
|`--parameters <string>`|`-p`|XGBoost parameters used for each model in the chain. Example: `-p "{'silent':1, 'eta':0.1}"` (default: {})|no|
|`--features_to_transform <string>`|`-t`|A list of all features in the dataset that have to be encoded. XGBoost can only process numerical features. Use this parameter to encode categorical features. Example: `-t "featureA,featureB"`|no|
|`--output_extra`|`-o`|Write extended log and json files (default: True)|no|

## Example

We train two models, the dynamic chain and a binary relevance model, on a dataset called `emotions`  with 6 labels. So we specify the models with `-m "dc, br"` and the dataset with `-f "emotions"`. Additionally we place the files for training and testing into the datasets directory:
```
project
│   README.md
│   train_dcc.py   
│
└───datasets
│   │   emotions-train.arff
│   │   emotions-test.arff
│   
└───XGBoost_ML
    │   ...

```

The dcc model should build a full chain with 6 models, so we use `-l 6`. All XGBoost models, also the one for binary relevance, should train for 100 rounds with a maximum tree depth of 10 and a step size of 0.1. Therefore we add `-p "{'eta':0.1}" -r 100 -d 10`

The full command to train and evaluate both models is:

     train_dcc.py -p "{'eta':0.1}" -f "emotions" -l 6 -r 100 -d 10 -c 6 -m 'dcc, br'
     
    
   

