#!/bin/bash

cd $1 && accelerate launch --num_processes 16 --num_machines 4 --main_process_ip $2 --main_process_port $3 --machine_rank $4 $1/llm-ddp-train.py