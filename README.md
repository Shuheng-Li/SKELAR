# SKELAR
The code base of paper **Matching Skeleton-based Activity Representations with Heterogeneous Signals for HAR** (ear in SenSys 2025).

## Data
The dataset used in the paper is MASD (**M**ultimodal **A**ctivity **S**ensing **D**ataset) shared in UCSD's [DataPlanet Platform](https://dataplanet.ucsd.edu/dataverse/masd)
The processed data will also be shared on Google Drive.

## Pretrain Code
In folder pretrain, run the following code to train the skeleton encoder, and get the label representations.
```
python train_autoencoder.py --decoder angle
python get_masd_embedding.py 
```

## Downstream Code
With pretrained label weights, at each task folder in the downstream folder run `python main.py` to use the label representation for downstream HAR.

## In Construction
More detailed instruction and scripts will be uploaded to the repo soon!
