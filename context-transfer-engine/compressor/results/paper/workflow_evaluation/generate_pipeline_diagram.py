#!/usr/bin/env python3
"""
Generate pipeline efficiency diagram comparing DTSchedule vs. HCompress.
Shows how DTSchedule enables pipelining while HCompress blocks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def create_pipeline_diagram():
    """Create pipeline efficiency visualization."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Time constants (in milliseconds, scaled for visualization)
    sim_time = 1.0
    compress_time = 40.0  # Scaled down from 40,320ms for visualization
    network_time = 0.077  # 77ms scaled

    # Colors
    color_sim = '#2ecc71'
    color_compress = '#e74c3c'
    color_network = '#3498db'
    color_idle = '#ecf0f1'

    # HCompress: Sequential execution (no pipelining)
    ax1.set_title('HCompress: Sequential Execution (No Pipelining)',
                  fontsize=14, fontweight='bold', pad=15)

    y_producer = 1.5
    y_consumer = 0.5

    # Iteration 1
    t = 0
    ax1.barh(y_producer, sim_time, left=t, height=0.3,
             color=color_sim, edgecolor='black', label='Simulation')
    t += sim_time
    ax1.barh(y_producer, compress_time, left=t, height=0.3,
             color=color_compress, edgecolor='black', label='Compression')
    t += compress_time

    # Iteration 2
    ax1.barh(y_producer, sim_time, left=t, height=0.3,
             color=color_sim, edgecolor='black')
    t += sim_time
    ax1.barh(y_producer, compress_time, left=t, height=0.3,
             color=color_compress, edgecolor='black')
    t += compress_time

    # Iteration 3
    ax1.barh(y_producer, sim_time, left=t, height=0.3,
             color=color_sim, edgecolor='black')
    t += sim_time
    ax1.barh(y_producer, compress_time, left=t, height=0.3,
             color=color_compress, edgecolor='black')

    # Consumer is idle for HCompress
    total_time_hcompress = (sim_time + compress_time) * 3
    ax1.barh(y_consumer, total_time_hcompress, left=0, height=0.3,
             color=color_idle, edgecolor='black', alpha=0.5, label='Idle')

    ax1.set_ylim(0, 2.5)
    ax1.set_xlim(0, total_time_hcompress)
    ax1.set_yticks([y_consumer, y_producer])
    ax1.set_yticklabels(['Consumer', 'Producer'], fontsize=12)
    ax1.set_xlabel('Time (scaled ms)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, ncol=4)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add total time annotation
    ax1.text(total_time_hcompress / 2, 2.2,
             f'Total Time: {total_time_hcompress:.1f} scaled ms\n(3 iterations)',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # DTSchedule: Pipelined execution
    ax2.set_title('DTSchedule: Pipelined Execution (Compression Offloaded)',
                  fontsize=14, fontweight='bold', pad=15)

    # Producer pipeline
    t = 0
    for i in range(3):
        ax2.barh(y_producer, sim_time, left=t, height=0.3,
                 color=color_sim, edgecolor='black', label='Simulation' if i == 0 else '')
        t += sim_time
        ax2.barh(y_producer, network_time, left=t, height=0.3,
                 color=color_network, edgecolor='black', label='Network Transfer' if i == 0 else '')
        t += network_time

    total_time_dtschedule = (sim_time + network_time) * 3

    # Consumer pipeline (starts after first data arrives)
    t_consumer = sim_time + network_time
    for i in range(3):
        if t_consumer < total_time_dtschedule:
            ax2.barh(y_consumer, compress_time, left=t_consumer, height=0.3,
                     color=color_compress, edgecolor='black', alpha=0.8,
                     label='Compression (Offloaded)' if i == 0 else '')
            t_consumer += compress_time

    # Fill idle time before first data arrives
    ax2.barh(y_consumer, sim_time + network_time, left=0, height=0.3,
             color=color_idle, edgecolor='black', alpha=0.5, label='Idle')

    ax2.set_ylim(0, 2.5)
    ax2.set_xlim(0, max(total_time_dtschedule, t_consumer))
    ax2.set_yticks([y_consumer, y_producer])
    ax2.set_yticklabels(['Consumer', 'Producer'], fontsize=12)
    ax2.set_xlabel('Time (scaled ms)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11, ncol=5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add total time annotation
    ax2.text(total_time_dtschedule / 2, 2.2,
             f'Producer Total Time: {total_time_dtschedule:.2f} scaled ms\n(3 iterations)',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Add speedup annotation
    speedup = total_time_hcompress / total_time_dtschedule
    fig.text(0.5, 0.02,
             f'Speedup: {speedup:.1f}× (Producer completes {speedup:.1f}× faster with DTSchedule)',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    return fig

def main():
    """Generate and save pipeline diagram."""
    output_dir = Path(__file__).parent

    print("Generating pipeline efficiency diagram...")

    fig = create_pipeline_diagram()

    # Save
    output_path = output_dir / 'pipeline_efficiency.pdf'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    output_path_svg = output_dir / 'pipeline_efficiency.svg'
    fig.savefig(output_path_svg, bbox_inches='tight')
    print(f"Saved: {output_path_svg}")

    plt.close()

    print("✓ Pipeline diagram complete!")

if __name__ == '__main__':
    main()
