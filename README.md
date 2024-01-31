# Code for "Beyond behaviour change: investigating alternative explanations for shorter time headways when human drivers follow automated vehicles"

# Abstract
Integrating Automated Vehicles (AVs) into existing traffic systems holds the promise of enhanced road safety, reduced congestion, and more sustainable travel. Effective integration of AVs requires understanding the interactions between AVs and Human-driving Vehicles (HVs), especially during the transition period in which AVs and HVs coexist in a mixed traffic environment. Numerous recent empirical studies find reduced headways of human drivers following an AV compared to following an HV, and attribute this reduction to behaviour changes of following vehicle drivers. However, more factors may be at play due to the inherent inconsistencies between the comparison conditions of HV-following-AV and HV-following-HV. This study scrutinises three alternative explanations for the observed reduction in headways: (1) systematic differences in car-following states during data collection, (2) systematic differences in driving variability between leading AVs and HVs, and (3) systematic differences in driving characteristics of leading AVs versus HVs. We use a large-scale dataset extracted from Lyft AV motion data, and isolate each of these explanations by data stratification and simulation. Our results show that all three mechanisms contribute to the observed reduction in headways of human drivers following AVs. In addition, our findings highlight the importance of driving homogeneity and stability in achieving reliably shorter headways. Thereby, this study offers a more comprehensive understanding on the difference between HV-AV and HV-HV interactions in mixed traffic, and is expected to promote more effective integration of AVs into human traffic.

## Package requirements
`zarr`, `numpy`, 'pandas`, `tqdm`, `matplotlib`, `scipy`, `joblib`, `pytorch`

## In order to repeat the experiments:

1. Download and save data

Download raw data of Lyft from <https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV> and save them in the folder ; download raw data of Lyft from <https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV> and save them in the folder "Data path example/InputData/Waymo/".

__Step 2.__ Run `Preprocessing.py` in the folder `Code` first to preprocess the rawdata.

__Step 3.__ Use `IntersectionDetection.py` and `IntersectionData.ipynb` in the folder `Code` to identify and select intersections in the pNEUMA dataset.

__Step 4.__ Run `Sampling_exp1-2.py`, `Sampling_exp3.ipynb`, and `Sampling_exp4.py` in the folder `Code` to transform coordinates, and sample vehicle pairs for different experiments.

__Step 4.__ Run `Experiments.py` to repeat our experiments in the article.

__*__ `DriverSpaceInference.py` is the library including classes and functions for the experiments

__*__ We run the experiments in Linux with a cluster of CPUs. To be run on other OSs may need adjustments regarding the number of cores for parallel processing.

## In order to apply the method to another dataset:

__Step 1.__ Save raw data in the folder "RawDatasets".

__Step 2.__ Create code to align the format of the new dataset to the format of the data to be saved in the folder "InputData".

__Step 3.__ Design your application according to the code in `Experiments.py`.

# Citation
````latex
@article{Jiao2023,
  doi = {10.1016/j.trc.2023.104289},
  year = {2023},
  month = oct,
  publisher = {Elsevier {BV}},
  volume = {155},
  pages = {104289},
  author = {Yiru Jiao and Simeon C. Calvert and Sander {van Cranenburgh} and Hans {van Lint}},
  title = {Inferring vehicle spacing in urban traffic from trajectory data},
  journal = {Transportation Research Part C: Emerging Technologies}
}
````
