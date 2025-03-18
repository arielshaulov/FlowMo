import matplotlib.pyplot as plt
import re
import numpy as np

def parse_file(file_path):
    """Parse the variance data file and extract timesteps and variance values."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract data using regex
    timesteps = [int(match) for match in re.findall(r'In timestep: (\d+)', content)]
    motion_variance = [float(match) for match in re.findall(r'motion_variance: ([\d.]+)', content)]
    appearance_variance = [float(match) for match in re.findall(r'motion_appearance_variance: ([\d.]+)', content)]
    
    return timesteps, motion_variance, appearance_variance

def main():
    # File paths
    optimized_file = '/home/ai_center/ai_users/arielshaulov/Wan2.1/zero_var/A_figure_skater_gliding_gracefully_across_the_ice_150.txt'
    non_optimized_file = '/home/ai_center/ai_users/arielshaulov/Wan2.1/runs_without_opti/A_figure_skater_gliding_gracefully_across_the_ice_150.txt'
    
    # Parse files
    timesteps_opt, motion_var_opt, appearance_var_opt = parse_file(optimized_file)
    timesteps_non_opt, motion_var_non_opt, appearance_var_non_opt = parse_file(non_optimized_file)
    
    # Create figure with two subplots for better visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
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
    plt2, ax3 = plt.subplots(figsize=(12, 8))
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
    plt2.savefig('variance_comparison_combined.png', dpi=300, bbox_inches='tight')
    
    # plt.show()

if __name__ == "__main__":
    main()