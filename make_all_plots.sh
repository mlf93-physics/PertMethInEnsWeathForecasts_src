#!/bin/bash

# Make shell model plots
cd ./shell_model_experiments

echo "Making sh_howmoller_vs_time, sh_energy_vs_time and sh_energy_spectrum plot"
python ./plotting/collect_plots.py --plot_type=spec_energy_howmoller --ref_end_time=1 --tolatex -lf three_panel --save_fig --noplot

echo "Making sh_eigenmode_analysis plot"
python ./plotting/collect_plots.py --plot_type=lyapunov_anal --n_profiles=100 --tolatex --save_fig --noplot --noconfirm

# L63 plots
cd ./lorentz63_experiments

echo "Make 3D pert method visualization"
python plotting/compare_plot.py --n_profiles=1 --plot_type=pert_vectors3D --exp_folder=compare_pert_3dplot_with_attractor -v bv_eof bv sv lv -pt rd nm rf  --seed_mode --file_offset=0 --tolatex -lf two_quads --save_fig --save_fig_name=pert_vectors_3D_v1 --noplot plot_kwargs --elev=2 --azim=-118
python plotting/compare_plot.py --n_profiles=1 --plot_type=pert_vectors3D --exp_folder=compare_pert_3dplot_with_attractor -v bv_eof bv sv lv -pt rd nm rf  --seed_mode --file_offset=0 --tolatex -lf two_quads --save_fig --save_fig_name=pert_vectors_3D_v2 --noplot plot_kwargs --elev=9 --azim=-32

echo "Make compare error norm plot"
python ../general/plotting/collect_plot_comparison.py --plot_type=collect_error_norm_compare --exp_folder=compare_pert_error_norm --endpoint  --seed_mode --n_runs_per_profile=30 --tolatex -lf two_panel --right_spine --notight --save_fig --noplot