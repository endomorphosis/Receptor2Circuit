"""Example usage of Allen Mouse Connectivity Loader.

This script demonstrates how to load and use connectivity data from the
Allen Mouse Brain Connectivity Atlas.
"""

import numpy as np
import matplotlib.pyplot as plt

from neurothera_map.mouse.allen_connectivity import (
    AllenConnectivityLoader,
    load_allen_connectivity,
)


def example_basic_loading():
    """Example 1: Basic connectivity loading with visual cortex regions."""
    print("Example 1: Loading connectivity for visual cortex regions")
    print("-" * 60)
    
    # Define regions of interest (visual cortex areas)
    regions = ['VISp', 'VISl', 'VISal', 'VISrl', 'VISam', 'VISpm']
    
    try:
        # Load connectivity matrix
        connectivity = load_allen_connectivity(
            region_acronyms=regions,
            normalize=True,
            threshold=0.01  # Filter weak connections
        )
        
        print(f"Loaded connectivity for {len(connectivity.region_ids)} regions")
        print(f"Regions: {', '.join(connectivity.region_ids)}")
        print(f"Adjacency matrix shape: {connectivity.adjacency.shape}")
        print(f"Number of non-zero connections: {np.count_nonzero(connectivity.adjacency)}")
        print(f"Mean connection strength: {connectivity.adjacency[connectivity.adjacency > 0].mean():.4f}")
        print()
        
        return connectivity
        
    except ImportError:
        print("ERROR: Allen SDK not installed. Install with: pip install allensdk")
    except Exception as e:
        print(f"ERROR: Could not load connectivity: {e}")
    
    return None


def example_advanced_loading():
    """Example 2: Advanced usage with the AllenConnectivityLoader class."""
    print("Example 2: Advanced usage with AllenConnectivityLoader")
    print("-" * 60)
    
    try:
        # Initialize loader
        loader = AllenConnectivityLoader(resolution=25)
        
        # Get available structures
        structures = loader.get_available_structures()
        print(f"Total available structures: {len(structures)}")
        
        # Filter to cortical regions
        cortical = structures[structures['acronym'].str.contains('VIS|MOp|SSp', na=False)]
        print(f"Cortical structures found: {len(cortical)}")
        
        # Load connectivity for motor and somatosensory cortex
        regions = ['MOp', 'MOs', 'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul']
        
        connectivity = loader.load_connectivity_matrix(
            region_acronyms=regions,
            normalize=True,
            threshold=0.0
        )
        
        print(f"\nLoaded connectivity for: {', '.join(connectivity.region_ids)}")
        print(f"Provenance: {connectivity.provenance}")
        print()
        
        return connectivity
        
    except ImportError:
        print("ERROR: Allen SDK not installed. Install with: pip install allensdk")
    except Exception as e:
        print(f"ERROR: Could not load connectivity: {e}")
    
    return None


def example_visualization(connectivity):
    """Example 3: Visualize connectivity matrix."""
    print("Example 3: Visualizing connectivity matrix")
    print("-" * 60)
    
    if connectivity is None:
        print("No connectivity data to visualize")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot adjacency matrix
        im = ax.imshow(connectivity.adjacency, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(connectivity.region_ids)))
        ax.set_yticks(range(len(connectivity.region_ids)))
        ax.set_xticklabels(connectivity.region_ids, rotation=45, ha='right')
        ax.set_yticklabels(connectivity.region_ids)
        
        ax.set_xlabel('Target Region')
        ax.set_ylabel('Source Region')
        ax.set_title('Allen Mouse Brain Connectivity Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Projection Strength')
        
        plt.tight_layout()
        
        # Save figure
        output_path = 'allen_connectivity_matrix.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        
        # Optionally display
        # plt.show()
        
    except Exception as e:
        print(f"ERROR: Could not create visualization: {e}")


def example_network_analysis(connectivity):
    """Example 4: Basic network analysis of connectivity."""
    print("\nExample 4: Network analysis")
    print("-" * 60)
    
    if connectivity is None:
        print("No connectivity data to analyze")
        return
    
    # Out-degree (total outgoing connections)
    out_degree = connectivity.adjacency.sum(axis=1)
    
    # In-degree (total incoming connections)
    in_degree = connectivity.adjacency.sum(axis=0)
    
    print("Out-degree (outgoing connections) by region:")
    for region, degree in zip(connectivity.region_ids, out_degree):
        print(f"  {region}: {degree:.4f}")
    
    print("\nIn-degree (incoming connections) by region:")
    for region, degree in zip(connectivity.region_ids, in_degree):
        print(f"  {region}: {degree:.4f}")
    
    # Identify hubs (regions with high connectivity)
    mean_out_degree = out_degree.mean()
    hubs = [region for region, degree in zip(connectivity.region_ids, out_degree) 
            if degree > mean_out_degree]
    
    print(f"\nHub regions (out-degree > mean): {', '.join(hubs)}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Allen Mouse Connectivity Loader - Examples")
    print("=" * 60)
    print()
    
    # Run examples
    connectivity1 = example_basic_loading()
    connectivity2 = example_advanced_loading()
    
    # Use the first successful connectivity for visualization
    connectivity = connectivity1 if connectivity1 is not None else connectivity2
    
    if connectivity is not None:
        example_visualization(connectivity)
        example_network_analysis(connectivity)
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
