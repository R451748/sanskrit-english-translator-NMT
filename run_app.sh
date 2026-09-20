#!/bin/bash

cd /mnt/c/Users/rohan/MTech_Sanskrit_NMT

source ~/indictrans-env/bin/activate

streamlit run application/app.py --server.headless true &

sleep 5

cmd.exe /c start http://localhost:8501

wait
