import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Generate and visualize POI and base station positions, save to .npy file")
    parser.add_argument('--n_poi', type=int, default=20, help="Number of POIs")
    parser.add_argument('--n_bs', type=int, default=1, help="Number of base stations")
    parser.add_argument('--map_width', type=float, default=1000.0, help="Map width")
    parser.add_argument('--map_height', type=float, default=1000.0, help="Map height")
    parser.add_argument('--output_dir', type=str, default="./data", help="Output directory for .npy and .png files")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument('--visualize', action='store_true', default=True, 
                        help="Generate and save a visualization of the positions (default: True)")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for the visualization image")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_poi < 0 or args.n_bs < 0:
        raise ValueError("Number of POIs and base stations must be non-negative")
    if args.map_width <= 0 or args.map_height <= 0:
        raise ValueError("Map width and height must be positive")
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    
    return args

def generate_positions(n_poi, n_bs, map_width, map_height, seed=None):
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random positions
    poi_positions = np.random.uniform(low=0.0, high=[map_width, map_height], size=(n_poi, 2))
    bs_positions = np.random.uniform(low=0.0, high=[map_width, map_height], size=(n_bs, 2))
    
    return poi_positions, bs_positions

def visualize_positions(poi_positions, bs_positions, map_width, map_height, output_dir, n_poi, n_bs, dpi):
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Plot POIs (blue circles)
    if poi_positions.size > 0:
        plt.scatter(poi_positions[:, 0], poi_positions[:, 1], c='blue', marker='o', label='POIs', alpha=0.6)
    
    # Plot base stations (red triangles)
    if bs_positions.size > 0:
        plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', s=100, label='Base Stations', alpha=0.8)
    
    # Set plot properties
    plt.xlim(0, map_width)
    plt.ylim(0, map_height)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'POI and Base Station Positions\n({n_poi} POIs, {n_bs} Base Stations, Map: {int(map_width)}x{int(map_height)})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save plot
    filename = f"poi_{n_poi}_bs_{n_bs}_map_{int(map_width)}x{int(map_height)}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filepath}")

def save_positions(poi_positions, bs_positions, n_poi, n_bs, map_width, map_height, output_dir, visualize, dpi):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save positions to .npy file
    filename = f"poi_{n_poi}_bs_{n_bs}_map_{int(map_width)}x{int(map_height)}.npy"
    filepath = os.path.join(output_dir, filename)
    data = {
        'poi_positions': poi_positions,
        'bs_positions': bs_positions
    }
    np.save(filepath, data)
    print(f"Saved positions to {filepath}")
    
    # Generate visualization if enabled
    if visualize:
        visualize_positions(poi_positions, bs_positions, map_width, map_height, output_dir, n_poi, n_bs, dpi)

def main():
    args = parse_args()
    
    # Generate positions
    poi_positions, bs_positions = generate_positions(
        args.n_poi, args.n_bs, args.map_width, args.map_height, args.seed
    )
    
    # Save positions and optionally visualize
    save_positions(
        poi_positions, bs_positions, args.n_poi, args.n_bs, 
        args.map_width, args.map_height, args.output_dir, args.visualize, args.dpi
    )

if __name__ == "__main__":
    main()