#!/usr/bin/env bash

echo "begin exp"
python src/Self_Consistency.py
wait
python src/Self_Consistency.py
wait
python src/Self_Consistency.py
wait
python src/Self_Consistency.py


echo "End for svamp"