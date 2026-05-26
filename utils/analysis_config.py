
from typing import Dict, List, Optional
from dataclasses import dataclass
import multiprocessing as mp

@dataclass
class AnalysisConfig:

    baseline_params: Dict[str, float] = None
    parameter_ranges: Dict[str, List[float]] = None
    boundary_conditions: List[str] = None
    distribution_types: List[str] = None
    base_output_dir: str = "results"
    script_name: str = "sensitivity_analysis"

    max_workers: Optional[int] = None
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    resource_limit: float = 0.7
    min_free_memory_gb: float = 4.0
    min_free_cores: int = 2
    enable_resource_monitor: bool = True
    cpu_threshold: int = 80
    memory_threshold: int = 85

    def __post_init__(self):

        if self.baseline_params is None:
            self.baseline_params = {
                'W_Gr': 0.025,
                'H_Gr': 0.8,
                'T': 300,
                'q': -0.08,
                'L_factor': 20,
                'h': 0.1,
                'num_layers': 10,
                'k1': 0.0,
                'k2': 0.0,
                'A0': 0.0,
                'a': 0.0,
                'b': 0,
                'c': 0.5,
                'bc_type': 'C-C',
                'distribution': 'X'
            }

        if self.parameter_ranges is None:
            self.parameter_ranges = {
                'W_Gr': [0, 0.005, 0.010, 0.015, 0.020, 0.025],
                'H_Gr': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'T': [300, 325, 350, 375, 400],
                'q': [0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12, -0.13, -0.14, -0.15, -0.16, -0.17, -0.18, -0.19, -0.20],
                'L_factor': [10, 20, 30, 40, 50],
                'k1': [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05],
                'k2': [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
            }

        if self.boundary_conditions is None:
            self.boundary_conditions = ['C-C', 'C-H', 'H-H', 'S-S', 'C-F']

        if self.distribution_types is None:
            self.distribution_types = ['X', 'U', 'O']

        if self.max_workers is None:
            self.max_workers = max(1, int(mp.cpu_count() * 0.7))

    def get_total_cases(self) -> int:
        total_params = sum(len(values) for values in self.parameter_ranges.values())
        total_combinations = len(self.boundary_conditions) * len(self.distribution_types)
        return total_params * total_combinations

    def validate(self) -> bool:
        if not self.baseline_params or not self.parameter_ranges:
            return False

        if not self.boundary_conditions or not self.distribution_types:
            return False

        if self.max_workers is not None and self.max_workers < 1:
            return False

        return True

    def print_summary(self):
        print("\n" + "="*60)
        print("Parameter Sensitivity Analysis Configuration Summary")
        print("="*60)
        print(f"Script name: {self.script_name}")
        print(f"Output directory: {self.base_output_dir}")
        print(f"Boundary conditions: {len(self.boundary_conditions)} types {self.boundary_conditions}")
        print(f"Distribution types: {len(self.distribution_types)} types {self.distribution_types}")
        print(f"Analysis parameters: {len(self.parameter_ranges)} params {list(self.parameter_ranges.keys())}")
        print(f"Total cases: {self.get_total_cases()}")

        if self.max_workers is not None:
            print(f"\nParallel configuration:")
            print(f"  Workers: {self.max_workers}")
            print(f"  Resource limit: {self.resource_limit:.1%}")
            print(f"  GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
            if self.enable_resource_monitor:
                print(f"  Resource monitor: Enabled (CPU<{self.cpu_threshold}%, Memory<{self.memory_threshold}%)")

        print("="*60)

PARAM_LABELS = {
    'W_Gr': ('Graphene Mass Fraction W_Gr', 'W_Gr'),
    'H_Gr': ('Graphene Shape Factor H_Gr', 'H_Gr'),
    'T': ('Temperature (K)', 'Temperature'),
    'q': ('Load Magnitude |q|', 'Load'),
    'L_factor': ('Length Ratio L/h', 'Length Ratio'),
    'k1': ('Winkler Foundation Stiffness k1', 'k1'),
    'k2': ('Pasternak Foundation Stiffness k2', 'k2')
}

def create_standard_config(script_name: str = "sensitivity_analysis",
                          enable_parallel: bool = False) -> AnalysisConfig:
    config = AnalysisConfig(script_name=script_name)

    if not enable_parallel:
        config.max_workers = None
        config.enable_resource_monitor = False

    return config
