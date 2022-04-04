#!/bin/bash

# Make shell model plots
cd ./shell_model_experiments

echo "Making sh_howmoller_vs_time, sh_energy_vs_time and sh_energy_spectrum plot"
python ./plotting/collect_plots.py --plot_type=spec_energy_howmoller --ref_end_time=1 --tolatex -lf three_panel --save_fig --noplot

echo "Making sh_eigenmode_analysis plot"
python ./plotting/collect_plots.py --plot_type=lyapunov_anal --n_profiles=100 --tolatex --save_fig --noplot --noconfirm

echo "Making error norm comparison"
python ../general/plotting/collect_plot_comparison.py --plot_type=collect_error_norm_compare --exp_folder=compare_pert_error_norm --endpoint  --seed_mode -pt all --tolatex -lf two_panel --save_fig --right_spine --notight plot_kwargs

echo "Make exp. growth rate comparison"
python ../general/plotting/collect_plot_comparison.py --plot_type=collect_exp_growth_rate_compare_plots --endpoint --seed_mode -pt all --n_runs_per_profile=20 --tolatex -lf two_panel --save_fig --noplot --noconfirm plot_kwargs --exp_growth_type=instant

echo "Making singular vector average plot, opt0.01"
python ../general/plotting/singular_vector_plotter.py --plot_type=s_vectors_average --pert_vector_folder=low_pred/compare_pert_exp_growth_rate_it0.01/vectors --exp_folder=sv_vectors --endpoint --n_profiles=1000 --n_runs_per_profile=20 --tolatex -lf quad_item --notight --save_fig --noconfirm --save_sub_folder=results_and_analyses/shell/low_pred --noplot

echo "Making singular vector average plot, opt0.01"
python ../general/plotting/singular_vector_plotter.py --plot_type=s_vectors_average --pert_vector_folder=low_pred/compare_pert_exp_growth_rate_it0.005/vectors --exp_folder=sv_vectors --endpoint --n_profiles=1000 --n_runs_per_profile=20 --tolatex -lf quad_item --notight --save_fig --noconfirm --save_sub_folder=appendices/extra_plots/shell_opt0.005/low_pred --noplot

# L63 plots
cd ./lorentz63_experiments

echo "Make BV vs LV projectibility plot"
python ../general/plotting/collect_plot_comparison.py --plot_type=collect_bv_vec_compare_plots --exp_folder=compare_vector_projectibility_to_lv_n_units1000 --endpoint --n_runs_per_profile=3 --n_profiles=1000 -nlvs=3 --tolatex -lf normal_small --save_fig --noplot --noconfirm

echo "Make SV/FSV vs LV/ALV projectibility plot"
python ../general/plotting/collect_plot_comparison.py --plot_type=collect_sv_vec_compare_plots --exp_folder=compare_vector_projectibility_to_lv_n_units1000 --endpoint --n_runs_per_profile=3 --n_profiles=10 -nlvs=3 --tolatex -lf normal_small --notight --save_fig --noconfirm --noplot

echo "Make 3D pert method visualization"
python plotting/compare_plot.py --n_profiles=1 --plot_type=pert_vectors3D --exp_folder=compare_pert_3dplot_with_attractor -v bv_eof bv sv lv -pt rd nm rf  --seed_mode --file_offset=0 --tolatex -lf large_quad --save_fig --save_fig_name=pert_vectors_3D_v1 --noplot plot_kwargs --elev=2 --azim=-118
python plotting/compare_plot.py --n_profiles=1 --plot_type=pert_vectors3D --exp_folder=compare_pert_3dplot_with_attractor -v bv_eof bv sv lv -pt rd nm rf  --seed_mode --file_offset=0 --tolatex -lf large_quad --save_fig --save_fig_name=pert_vectors_3D_v2 --noplot plot_kwargs --elev=9 --azim=-32

echo "Make compare error norm plot"
python ../general/plotting/collect_plot_comparison.py --plot_type=collect_error_norm_compare --exp_folder=compare_pert_error_norm --endpoint  --seed_mode --n_runs_per_profile=30 --tolatex -lf two_panel --right_spine --notight --save_fig --noplot

echo "Make compare exp. growth rate dists plot"
python plotting/compare_plot.py  --plot_type=growth_rate_dist --exp_folder=compare_pert_exp_growth_rate_dists --seed_mode --n_runs_per_profile=3 -pt all --tolatex --notight -lf normal_large --save_fig --noplot plot_kwargs --exp_growth_type=mean

echo "Make compare exp. growth rates plot"
python ../general/plotting/plot_comparisons.py --plot_type=exp_growth_rate_compare --exp_folder=compare_pert_exp_growth_rate --endpoint  --seed_mode --n_runs_per_profile=3 -pt all -lf normal_small --tolatex --save_fig --noplot plot_kwargs --exp_growth_type=instant

echo "Make conceptual visualisations"
python ../general/plotting/conceptual_visualizations.py --plot_type=breed_method -lf horizontal_panel --tolatex --save_fig --noplot

echo "Make l63_pert_vectors_on_trajectory plots"
python plotting/compare_plot.py  --plot_type=plot_pert_vector_dists --exp_folder=compare_pert_vectors_on_trajectory -v bv_eof lv sv --n_profiles=140 --endpoint --tolatex -lf full_page --noplot --save_fig --save_fig_name=l63_pert_vectors_on_trajectory_view1 plot_kwargs --elev=21 --azim=-57
python plotting/compare_plot.py  --plot_type=plot_pert_vector_dists --exp_folder=compare_pert_vectors_on_trajectory -v bv_eof lv sv --n_profiles=140 --endpoint --tolatex -lf full_page --noplot --save_fig --save_fig_name=l63_pert_vectors_on_trajectory_view2 plot_kwargs --elev=6 --azim=-126