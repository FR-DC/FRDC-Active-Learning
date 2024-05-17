# FRDC-Active-Learning

This repository documents all baseline findings for our SSL and AL experiments using the Dec 2020/May 2021 Chestnut dataset. We experimented with 9 species. This repository serves as a code compilation of my (Benjamin Chew) URECA-FYP project under Prof Francis Lee and Dr. JJ Sit.

# SSL 

Our SSL baseline is attained using the [Mixmatch algorithm](https://arxiv.org/abs/1905.02249). The SSL script can be found in MixMatch/mixmatch/semisl.py.

We have also provided comparisons of SSL to the standard SL approach. The SL script can likewise be found in MixMatch/mixmatch/sl.py.

# AL

Our AL baselines uses diversity, uncertainty and random sampling. We also implement a hybrid approach using our best performing diversity and uncertainty sampling strategy in conjunction with random sampling in a 10-10-80 ratio (diversity-uncertainty-random respectively). The sampling strategies have been implemented in MixMatch/mixmatch/utils/sampling.py.

To reproduce our experiments of AL, the AL script can be found in MixMatch/mixmatch/al.py.

# Dataset preparation 

To facilitate ease of experimentation and verification/reproduction of earlier results, we have uploaded the processed datasets. Each dataset folder is labelled as `chestnut_{date}_{grid_cell_size}_{optional_number_of_bands}_remote`. Note that there is an additional label of 43m in the May 2021 dataset naming convention after the `{date}` label. The `{optional_number_of_bands}` indicates the number of bands in the dataset, 3 for rgb and 8 if unspecified. Each dataset folder has 9 folders, each representing a different species. Each species folder contains the relevant `.npz` file for the specific grid in the image. 

Should there be a need to further generate data, the code under the FRDC-Loader/src/ folder will download the data from GCS and extract the relevant species. Run the data.py script under that folder to generate the core datasets. 

To perform further processing such as isolating the necessary bands of data or splitting the images into grid cells, run the script in MixMatch/mixmatch/utils/data_prep.py. This will generate the various folders as explained above. 
