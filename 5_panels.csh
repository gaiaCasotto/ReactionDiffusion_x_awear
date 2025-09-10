#!/bin/csh -fxv


mkdir -p  fifos
mkfifo fifos/awear_fifo1

python3 ./t_rd_colors_ac.py --port 5001 --title "User stressed" --file ../eeg_files/fake_eeg_longblocks.txt           >& awear_proto1_1.log &
python3 ./t_rd_colors_ac.py --port 5002 --title "User 2" --file ../eeg_files/fake_eeg_longblocks_calmfirst.txt >& awear_proto1_2.log &
python3 ./t_rd_colors_ac.py --port 5003 --title "User 3"  --file fifos/awear_fifo1 >& awear_proto1_3.log &


sleep 10
cat ../eeg_files/fake_eeg_longblocks.txt  > fifos/awear_fifo1
