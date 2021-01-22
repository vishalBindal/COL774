#!/bin/sh

python neural_network.py $1 $2 $3 $4 $5 "$6" $7
# Example:
# ./run_nn.sh "/home/vishal/X_train.npy" "/home/vishal/Y_train.npy" "/home/vishal/X_test.npy" "/home/vishal/2016CSZ8119_nn.txt" 100 "100 50" "softmax"
