# SKELAR
The code base of paper **Matching Skeleton-based Activity Representations with Heterogeneous Signals for HAR** (in SenSys 2025).

## Data
The dataset used in the paper is MASD (**M**ultimodal **A**ctivity **S**ensing **D**ataset) shared in UCSD's [DataPlanet Platform](https://dataplanet.ucsd.edu/dataverse/masd)
The raw data is available on [Google Drive](https://drive.google.com/drive/folders/1YElnaL0JNWG_Fj_xKmPQOpksd38Q2Vad?usp=drive_link) as a backup.

## Pretrain Code
In folder pretrain, run the following code to train the skeleton encoder, and get the label representations.
```
python train_autoencoder.py --decoder angle
python get_masd_embedding.py 
```

## Downstream Code
With pretrained label weights, at each task folder in the downstream folder run `python main.py` to use the label representation for downstream HAR.

