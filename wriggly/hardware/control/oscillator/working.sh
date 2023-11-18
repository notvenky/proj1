#!/bin/bash

# Set the default values
DEFAULT_COMMAND_FREQUENCY=3.0
DEFAULT_KEYS="11 12 20 21 22"

# Read user inputs
read -p "Enter COMMAND_FREQUENCY (default is $DEFAULT_COMMAND_FREQUENCY): " COMMAND_FREQUENCY
read -p "Enter Dynamixel IDs separated by spaces (default is $DEFAULT_KEYS): " KEYS

# If no value is entered by the user, use the default values
COMMAND_FREQUENCY=${COMMAND_FREQUENCY:-$DEFAULT_COMMAND_FREQUENCY}
KEYS=${KEYS:-$DEFAULT_KEYS}

# Replace the values in the Python file
sed -i "s/COMMAND_FREQUENCY = [0-9.]\+/COMMAND_FREQUENCY = $COMMAND_FREQUENCY/" filename.py
sed -i "s/keys = \[.*\]/keys = \[$KEYS\]/" filename.py

# Run the Python code using xvfb-run
xvfb-run python3 working_playaround.py