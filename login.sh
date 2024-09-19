#!/bin/bash

# You must change the value /path/to/susentry.py to the path of your
# python file.
export DISPLAY=:0.0
xhost +local:
python3 /path/to/susentry.py -l # run the python script for facial recognition
exit_status=$? # This grabs the exit status of the python script we just ran
if [ "${exit_status}" -ne 0 ]; # checks to see if exit status is anything other than 0
then
    echo "exit ${exit_status}"
    exit 1 # exit status 1 on python script fail (exit 1)
fi
echo "EXIT 0"
exit 0 # exit 0 if we get to this line