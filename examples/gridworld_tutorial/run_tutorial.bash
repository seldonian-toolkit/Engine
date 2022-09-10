#! /bin/bash
# Make sure to run "conda activate seldo" before running this script
python generate_data.py
python createSpec.py
python run_gridworld.py