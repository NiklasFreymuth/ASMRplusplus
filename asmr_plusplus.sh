# Stokes Ablations
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_general -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_dqn -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_architecture -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_fixed_penalty -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_adaptive_penalty -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_features -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_npde -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_global -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_reward -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_global -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_mapping -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_old -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_mapping -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_asmr_architecture -o -s --nocodecopy

## Stokes Baselines
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_flow_single_agent -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_flow_sweep -o -s --nocodecopy
# python main.py configs/asmr_journal_final/stokes_flow.yaml -e stokes_flow_vdgn -o -s --nocodecopy


# Laplace
# python main.py configs/asmr_journal_final/laplace.yaml -e laplace_asmr -o -s --nocodecopy
# python main.py configs/asmr_journal_final/laplace.yaml -e laplace_single_agent -o -s --nocodecopy
# python main.py configs/asmr_journal_final/laplace.yaml -e laplace_vdgn -o -s --nocodecopy
# python main.py configs/asmr_journal_final/laplace.yaml -e laplace_sweep -o -s --nocodecopy


# Poisson
# python main.py configs/asmr_journal_final/poisson.yaml -e poisson_asmr -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson.yaml -e poisson_asmr_generalizing -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson.yaml -e poisson_single_agent -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson.yaml -e poisson_vdgn -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson.yaml -e poisson_sweep -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson.yaml -e poisson_asmr_generalizing -o -s --nocodecopy


# Linear Elasticity
# python main.py configs/asmr_journal_final/linear_elasticity.yaml -e linear_elasticity_asmr -o -s --nocodecopy
# python main.py configs/asmr_journal_final/linear_elasticity.yaml -e linear_elasticity_single_agent -o -s --nocodecopy
# python main.py configs/asmr_journal_final/linear_elasticity.yaml -e linear_elasticity_vdgn -o -s --nocodecopy
# python main.py configs/asmr_journal_final/linear_elasticity.yaml -e linear_elasticity_sweep -o -s --nocodecopy


# Heat Diffusion
# python main.py configs/asmr_journal_final/heat_diffusion.yaml -e heat_diffusion_asmr -o -s --nocodecopy
# python main.py configs/asmr_journal_final/heat_diffusion.yaml -e heat_diffusion_single_agent -o -s --nocodecopy
# python main.py configs/asmr_journal_final/heat_diffusion.yaml -e heat_diffusion_vdgn -o -s --nocodecopy
# python main.py configs/asmr_journal_final/heat_diffusion.yaml -e heat_diffusion_sweep -o -s --nocodecopy


# Poisson3d - (scheduled 12.02.)
# python main.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_asmr -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_asmr_old -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_single_agent -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_vdgn -o -s --nocodecopy
# python main.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_sweep -o -s --nocodecopy

## heuristics - (done 05.02.)
# python asmr_evaluations/evaluate_zzerror_heuristic.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_no_max_elements -o -s --nocodecopy
# python asmr_evaluations/evaluate_uniform_mesh.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_no_max_elements -o -s --nocodecopy
# python asmr_evaluations/evaluate_oracle_heuristic.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_no_max_elements -o -s --nocodecopy
# python asmr_evaluations/evaluate_oracle_heuristic.py configs/asmr_journal_final/poisson3d.yaml -e poisson3d_oracle_maxheuristic -o -s --nocodecopy


# Neumann Boundary
#python main.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_asmr -o -s --nocodecopy
#python main.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_asmr_old -o -s --nocodecopy
#python main.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_single_agent -o -s --nocodecopy
#python main.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_vdgn -o -s --nocodecopy
#python main.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_sweep -o -s --nocodecopy

## heuristics
# python asmr_evaluations/evaluate_zzerror_heuristic.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_no_max_elements -o -s --nocodecopy
# python asmr_evaluations/evaluate_uniform_mesh.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_no_max_elements -o -s --nocodecopy
# python asmr_evaluations/evaluate_oracle_heuristic.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_no_max_elements -o -s --nocodecopy
# python asmr_evaluations/evaluate_oracle_heuristic.py configs/asmr_journal_final/neumann_poisson.yaml -e neumann_poisson_oracle_maxheuristic -o -s --nocodecopy

