#!/bin/bash

echo "alias python='python3'" >> /home/ubuntu/.bashrc
echo "export WANDB_API_KEY=<INSERT YOUR API KEY>" >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

git config --global user.name "UncoverAI" 
git config --global user.email "uncover.ai@bla.com"