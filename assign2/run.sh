#!/bin/bash

if [[ $1 == "1" ]]; then
    python3 q1.py $2 $3 $4
elif [[ $1 == "2" ]]; then
    python3 q2.py $2 $3 $4
elif [[ $1 == "3" ]]; then
    python3 q3.py $2 $3 $4
fi

