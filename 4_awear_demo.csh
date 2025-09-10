#!/bin/csh -fvx


python 5_t_rd.py --fs 256 --buffer-s 8 --port 5001 --user 1 --posx 50 --posy 50 >& demo1.log &
python 5_t_rd.py --fs 256 --buffer-s 8 --port 5002 --user 2 --posx 550 --posy 50>& demo2.log &
python 5_t_rd.py --fs 256 --buffer-s 8 --port 5003 --user 3 --posx 50 --posy 550>& demo3.log &
python 5_t_rd.py --fs 256 --buffer-s 8 --port 5004 --user 4 --posx 550 --posy 550>& demo4.log &

sleep 5



python client_eeg.py --host 127.0.0.1 --port 5001 --fs 256 --chunk 64 --duration 120 --demo-profile >& demo_client1.log &
python client_eeg.py --host 127.0.0.1 --port 5002 --fs 256 --chunk 64 --duration 120 --demo-profile >& demo_client2.log &
python client_eeg.py --host 127.0.0.1 --port 5003 --fs 256 --chunk 64 --duration 120 --demo-profile >& demo_client3.log &
python client_eeg.py --host 127.0.0.1 --port 5004 --fs 256 --chunk 64 --duration 120 --demo-profile >& demo_client4.log &

