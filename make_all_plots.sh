#!/bin/bash

# Make shell model plots
cd ./shell_model_experiments

echo "Making sh_howmoller_vs_time, sh_energy_vs_time and sh_energy_spectrum plot"
python ./plotting/collect_plots.py --plot_type=spec_energy_howmoller --ref_end_time=1 --tolatex -lf three_panel --save_fig --noplot

echo "Making sh_eigenmode_analysis plot"
python ./plotting/collect_plots.py --plot_type=lyapunov_anal --n_profiles=100 --tolatex --save_fig --noplot --noconfirm