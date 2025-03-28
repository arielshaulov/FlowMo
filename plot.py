import matplotlib.pyplot as plt
import re
import numpy as np
import config

def parse_file(file_path):
    if not file_path:
        return [], [], [], [], []
    """Parse the variance data file and extract timesteps and variance values."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract data using regex
    timesteps = [int(match) for match in re.findall(r'In timestep: (\d+)', content)]
    motion_variance = [float(match) for match in re.findall(r'motion_variance: ([\d.]+)', content)]
    appearance_variance = [float(match) for match in re.findall(r'motion_appearance_variance: ([\d.]+)', content)]
    mean_bb_motion = [float(match) for match in re.findall(r'mean best-buddy motion: ([\d.]+)', content)]
    mean_bb_loss = [0] + [float(match) for match in re.findall(r'mean best-buddy continuity loss: ([\d.]+)', content)]
    
    return timesteps, motion_variance, appearance_variance, mean_bb_motion, mean_bb_loss

def main():
    # File paths
    prompt = "A_figure_skater_gliding_gracefully_across_the_ice_150"
    project_dir = config.get('project_dir')
    optimized_file = f'{project_dir}/{prompt}.txt'
    non_optimized_file = '{project_dir}/runs_without_opti/{prompt}.txt'
    
    # Parse files
    timesteps_opt, motion_var_opt, appearance_var_opt, mean_bb_motion_opt, mean_bb_loss_opt = parse_file(optimized_file)
    timesteps_non_opt, motion_var_non_opt, appearance_var_non_opt, mean_bb_motion_non_opt, mean_bb_loss_non_opt = parse_file(non_optimized_file)
    
    # Create figure with two subplots for better visualization
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    
    # Plot motion variance
    ax1.plot(timesteps_opt, motion_var_opt, 'b-', label='Optimized (Motion)')
    ax1.plot(timesteps_non_opt, motion_var_non_opt, 'b--', label='Non-Optimized (Motion)')
    ax1.set_ylabel('Motion Variance')
    ax1.set_title('Comparison of Motion and Appearance Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Reverse x-axis
    ax1.invert_xaxis()
    
    # Plot appearance variance
    ax2.plot(timesteps_opt, appearance_var_opt, 'r-', label='Optimized (Appearance)')
    ax2.plot(timesteps_non_opt, appearance_var_non_opt, 'r--', label='Non-Optimized (Appearance)')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Motion Appearance Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Reverse x-axis (not needed as sharex=True, but included for clarity)
    ax2.invert_xaxis()
    
    # Adjust layout
    plt.tight_layout()
    
    # Alternative: If you want all data on a single plot instead of stacked subplots
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.plot(timesteps_opt, motion_var_opt, 'b-', label='Optimized (Motion)')
    ax3.plot(timesteps_non_opt, motion_var_non_opt, 'b--', label='Non-Optimized (Motion)')
    ax3.plot(timesteps_opt, appearance_var_opt, 'r-', label='Optimized (Appearance)')
    ax3.plot(timesteps_non_opt, appearance_var_non_opt, 'r--', label='Non-Optimized (Appearance)')
    
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Variance')
    ax3.set_title('Combined Plot: Motion and Appearance Variance Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Reverse x-axis
    ax3.invert_xaxis()
    
    # Save figures
    # fig.savefig('variance_comparison_stacked.png', dpi=300, bbox_inches='tight')
    fig2.savefig('variance_comparison_combined.png', dpi=300, bbox_inches='tight')
    
    # plot mean best buddy motion and continuity loss in subplots
    fig3, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax4.plot(timesteps_opt, mean_bb_motion_opt, 'b-', label='Optimized (BB Motion)')
    ax4.plot(timesteps_non_opt, mean_bb_motion_opt, 'b--', label='Non-Optimized (BB Motion)')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Mean Best-Buddies Motion')
    ax4.set_title('Mean Best-Buddies Motion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # Reverse x-axis
    ax4.invert_xaxis()

    # plt.show()

    ax5.plot(timesteps_opt, mean_bb_loss_opt, 'r-', label='Optimized (BB Loss)')
    ax5.plot(timesteps_non_opt, mean_bb_loss_non_opt, 'r--', label='Non-Optimized (BB Loss)')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Mean Best-Buddies Loss')
    ax5.set_title('Mean Best-Buddies Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    # Reverse x-axis
    ax5.invert_xaxis()

    fig3.savefig('bb_motion_and_loss.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()