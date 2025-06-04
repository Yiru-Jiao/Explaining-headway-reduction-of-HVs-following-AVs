# Beyond behaviour change: investigating alternative explanations for shorter time headways when human drivers follow automated vehicles
This study is published in the journal "Transportation Research Part C: Emerging Technologies" with gold open access, available at <https://doi.org/10.1016/j.trc.2024.104673>.

## Highlights
- Conflict detection involves a trade‐off between missed and false alarms.
- Probabilities of missed and false alarms are estimated from spacing distributions.
- Critical spacing is optimised to minimise missed and false alarms.
- Validation on synthetic and real-world conflicts confirms superior performance.
- Collision warning can be adaptive in varying traffic contexts and driver preferences.

## Package requirements
`jupyter notebook`, `zarr`, `numpy`, `pandas`, `pytables`, `tqdm`, `matplotlib`, `scipy`, `joblib`, `pytorch`

## In order to repeat the experiments:

__Step 0. Preparation__

Create a conda environment for repeating the experiments. Install the required packages as listed above.

Clone this repository, then either 1) create/define a folder for data saving and copy the subfolders in "Data"; or 2) use the folder "Data" directly.

__Step 1. Download and save data__

Download the trajectory data of Lyft from <https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV> and save them in the folder "Data/InputData/Lyft/"; download processed data of Waymo from <https://data.mendeley.com/datasets/wfn2c3437n/2> and save it (`all_seg_paired_cf_trj_final_with_large_vehicle.csv`) in the folder "Data/InputData/Waymo/".

__Step 2. Standardise data format (of Waymo and Lyft)__ 

Run `data_format_standardization.py` in the current parent folder first to preprocess the trajectories.

__Step 3. Regime categorisation__ 

In the folder "Car-following regime categorisation", use `regime categorisation.ipynb` to categorise car-following regimes in Waymo and Lyft datasets.

__Step 4. IDM calibration and simulation__ 

In the folder "Car-following modelling and simulation", run `idm_data_selection.py` to select car-following pairs that cover sufficient regimes for Intelligent Driver Modelling.

Then in the sub-folder "Car-following modelling and simulation/IDM calibration", run `idm_calibration.py` to calibrate IDMs and run `loss_computation.py` to calculate calibration loss. Further in the sub-folder "IDM calibration/Appendix", we offer the calibration of the other two car-following models, Newell and Gipps, to enhance the robustness of the results.

Finally, in the sub-folder "Car-following modelling and simulation/Controlled simulation", run `cross_follow_leader.py` and `cross_follow_follower.py` to simulate the designed experiments.

__Step 5. Leading vehicle classification__ 

In the folder "Leading vehicle driving classification", use `dataset_separation.py` to separate Lyft data into train, val, and test sets, and then use `classifier_lstm.ipynb` to train the LSTM classifier, validate the trained model, and save test results.

<br />

__*__ In doing regime categorisation, we resued the code from <https://github.com/slaypni/fastdtw> to apply fast dynamic time warping. This is also indicated in the folder.

__*__ We have run the IDM calibration in Linux with 15 CPUs. To be run on other OSs may need adjustments regarding the number of cores/workers for parallel processing.


## Citation
````latex
@article{Jiao2024,
  title = {Beyond behavioural change: Investigating alternative explanations for shorter time headways when human drivers follow automated vehicles},
  volume = {164},
  doi = {10.1016/j.trc.2024.104673},
  journal = {Transportation Research Part C: Emerging Technologies},
  author = {Jiao,  Yiru and Li,  Guopeng and Calvert,  Simeon C. and van Cranenburgh,  Sander and van Lint,  Hans},
  year = {2024},
  pages = {104673}
}
````
