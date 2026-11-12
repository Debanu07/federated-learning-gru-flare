# Federated Learning with FLWR and NVIDIA FLARE

This project implements Federated Learning using:

- Flower (FLWR)
- NVIDIA FLARE

Datasets used:

- MNIST
- FashionMNIST

## Project Overview

This repository compares:

1. Federated Learning using FLWR
2. Federated Learning using NVIDIA FLARE
3. Centralized training baseline using PyTorch

## Files

client.py → FLWR client  
server.py → FLWR server  

client_diff_flwr.py → FLWR client with different datasets  
server_diff_flwr.py → FLWR server with different datasets  

flare_server.py → FLARE job configuration  
flare_train.py → FLARE training script  

model.py → neural network architecture  

normal.py → centralized training baseline

## Run FLWR

Start server

python server.py

Start clients

python client.py 0  
python client.py 1

## Run FLARE

python flare_server.py

## Centralized Training

python normal.py