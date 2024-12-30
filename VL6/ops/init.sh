#!/bin/bash

echo "alias python='python3'" >> /home/ubuntu/.bashrc
echo "export WANDB_API_KEY=<INSERT YOUR API KEY>" >> /home/ubuntu/.bashrc
echo "export HF_TOKEN=<INSERT YOUR API TOKEN>" >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

git config --global user.name "YOUR NAME" 
git config --global user.email "YOUR.NAME@mail.com"