#!/bin/csh -xvf

python 5_t_rd.py --user 1  --eeg  ../eeg_files/1_horror_movie_data_filtered.txt >& debug1.log &
python 5_t_rd.py --user 2  --eeg  ../eeg_files/2_vipassana_data_filtered.txt    >& debug2.log &
python 5_t_rd.py --user 22  --eeg  ../eeg_files/3_hot_tub_data_filtered.txt      >& debug3.log &
#  ../eeg_files/fake_eeg_longblocks_calmfirst.txt
#  ../eeg_files/fake_eeg_longblocks.txt
