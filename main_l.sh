conda_path=/home/sjtu/anaconda3   # change to your own conda path

# Set ROS melodic
source /opt/ros/melodic/setup.bash

# Start Anaconda
# added by Anaconda3 5.3.1 installer
# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '$conda_path/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        . "$conda_path/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="$conda_path/bin:$PATH"
	#\export PYTHON_PATH="/home/sjtu/anaconda3/bin"
    fi
fi
unset __conda_setup
# <<< conda init <<<
conda deactivate
conda activate yolov5 # change to your conda environment
python main_l.py
