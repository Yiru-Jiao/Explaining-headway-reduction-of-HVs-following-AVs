# Code for "Beyond behaviour change: investigating alternative explanations for shorter time headways when human drivers follow automated vehicles"

A preprint is available at <https://ssrn.com/abstract=4713957 or http://dx.doi.org/10.2139/ssrn.4713957>.

## Abstract
Integrating Automated Vehicles (AVs) into existing traffic systems holds the promise of enhanced road safety, reduced congestion, and more sustainable travel. Effective integration of AVs requires understanding the interactions between AVs and Human-driving Vehicles (HVs), especially during the transition period in which AVs and HVs coexist in a mixed traffic environment. Numerous recent empirical studies find reduced headways of human drivers following an AV compared to following an HV, and attribute this reduction to behaviour changes of following vehicle drivers. However, more factors may be at play due to the inherent inconsistencies between the comparison conditions of HV-following-AV and HV-following-HV. This study scrutinises three alternative explanations for the observed reduction in headways: (1) systematic differences in car-following states during data collection, (2) systematic differences in driving variability between leading AVs and HVs, and (3) systematic differences in driving characteristics of leading AVs versus HVs. We use a large-scale dataset extracted from Lyft AV motion data, and isolate each of these explanations by data stratification and simulation. Our results show that all three mechanisms contribute to the observed reduction in headways of human drivers following AVs. In addition, our findings highlight the importance of driving homogeneity and stability in achieving reliably shorter headways. Thereby, this study offers a more comprehensive understanding on the difference between HV-AV and HV-HV interactions in mixed traffic, and is expected to promote more effective integration of AVs into human traffic.

## Package requirements
`jupyter notebook`, `zarr`, `numpy`, `pandas`, `tqdm`, `matplotlib`, `scipy`, `joblib`, `pytorch`

## In order to repeat the experiments:

__Step 0. Preparation__

Create a conda environment for repeating the experiments. Install the required packages as listed above.

Clone this repository, then either 1) create/define a folder for data saving and copy the subfolders in "Data path example"; or 2) use the folder "Data path example" directly.

__Step 1. Download and save data__

Download the trajectory data of Lyft from <https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV> and save them in the folder "Data path example/InputData/Lyft/"; download processed data of Waymo from <https://data.mendeley.com/datasets/wfn2c3437n/2> and save it (`all_seg_paired_cf_trj_final_with_large_vehicle.csv`) in the folder "Data path example/InputData/Waymo/".

__Step 2. Standardise data format (of Waymo and Lyft)__ 

Run `data_format_standardization.py` in the current parent folder first to preprocess the trajectories.

__Step 3. Regime categorisation__ 

In the folder "Car-following regime categorisation", use `regime categorisation.ipynb` to categorise car-following regimes in Waymo and Lyft datasets.

__Step 4. IDM calibration and simulation__ 

In the folder "Car-following modelling and simulation", run `idm_data_selection.py` to select car-following pairs that cover sufficient regimes for Intelligent Driver Modelling.

Then in the sub-folder "Car-following modelling and simulation/IDM calibration", run `idm_calibration.py` to calibrate IDMs and run `loss_computation.py` to calculate calibration loss.

Further in the sub-folder "Car-following modelling and simulation/Controlled simulation", run `cross_follow_leader.py` and `cross_follow_follower.py` to simulate the designed experiments.

__Step 5. Leading vehicle classification__ 

In the folder "Leading vehicle driving classification", use `dataset_separation.py` to separate Lyft data into train, val, and test sets, and then use `classifier_lstm.ipynb` to train the LSTM classifier, validate the trained model, and save test results.

<br />

__*__ In doing regime categorisation, we resued the code from <https://github.com/slaypni/fastdtw> to apply fast dynamic time warping. This is also indicated in the folder.

__*__ We have run the IDM calibration in Linux with 15 CPUs. To be run on other OSs may need adjustments regarding the number of cores/workers for parallel processing.


# Citation
````latex
@article{Jiao2024,
  doi = {},
  year = {2024},
  volume = {},
  pages = {},
  author = {Yiru Jiao and Guopeng Li and Simeon C. Calvert and Sander {van Cranenburgh} and Hans {van Lint}},
  title = {Beyond behaviour change: investigating alternative explanations for shorter time headways when human drivers follow automated vehicles},
  journal = {}
}
````
