

h = 0.1
L_factor = 20
L = L_factor * h
num_layers = 10

W_Gr = 0.025
H_Gr = 0.8
T = 300

q = -0.08

bc_type = 'C-C'
distribution = 'X'

lifting_basis = "poly"

k1 = 0.01
k2 = 0.001

A0 = 0.0
a = 0.0
b = 0
c = 0.5

epochs = 80000
lr = 8e-5
N_train = 10000
seed = 42

use_adaptive_lr = False
lr_early_max = 1e-3
lr_early_min = 2e-4
lr_mid_max = 2e-4
lr_mid_min = 1e-4
lr_late_fixed = 1e-4
lr_patience = 500
lr_improvement_threshold = 1e-6
lr_decay_factor = 0.5
lr_verbose = True

lr_warmup_epochs = 100
lr_early_ratio = 0.6
lr_mid_ratio = 0.85
lr_min_early_epochs = 1000
lr_min_mid_epochs = 5000

optimizer_type = 'Adam'
adamw_weight_decay = 1e-4
lbfgs_max_iter = 20
lbfgs_history_size = 50
lbfgs_line_search_fn = None

network_arch = 'shared'

activation_type = 'Tanh'

siren_omega_0 = 1.5
siren_omega_hidden = 1.5

enc_width = 128
enc_depth = 3
head_width = 64
head_depth = 2

encoder_dims_shared = [1, 32, 64, 128]
head_dims = [128, 64, 32, 1]
input_dim = 1

bc_weight = 1000.0

sampler = 'uniform'
sampler_reuse = True

integrator = 'agq'
agq_rule = 'G10K21'
agq_abs_tol = 1e-6
agq_rel_tol = 1e-4
agq_max_points = 4096
agq_max_depth = 100
agq_refine_every = 0
agq_fail_policy = 'use_partial'

ps_w_non_start = 1.0
ps_w_lin_start = 0.5
ps_cut_ratio = 0.8
ps_use_phi = False

use_pseudo_supervision = True

transfer_alpha = 0.3
transfer_ratio = 0.7
transfer_freq = 500
transfer_cut_ratio = 0.2

print_every = 1000
disable_gpu_monitor = True

generate_plots = True
plot_interval = 1000
save_best_model = True
save_model_interval = 5000

log_level = 'INFO'
log_interval = 100
verbose = True

csv_precision = 8
plot_dpi = 300
figure_format = 'png'

def validate_params():
    required_params = [
        'h', 'L', 'W_Gr', 'H_Gr', 'T', 'q', 'bc_type', 'distribution'
    ]

    global_vars = globals()
    missing_params = []

    for param in required_params:
        if param not in global_vars:
            missing_params.append(param)

    if missing_params:
        raise ValueError(f"缺少必需参数: {missing_params}")

    if h <= 0:
        raise ValueError("梁厚度h必须大于0")
    if L <= 0:
        raise ValueError("梁长度L必须大于0")
    if L_factor <= 0:
        raise ValueError("梁长厚比L_factor必须大于0")
    if num_layers < 1:
        raise ValueError("材料层数必须至少为1")
    if epochs <= 0:
        raise ValueError("训练轮数必须大于0")
    if lr <= 0:
        raise ValueError("学习率必须大于0")
    if N_train <= 0:
        raise ValueError("训练点数必须大于0")

    allowed_bc_types = {'C-C', 'C-H', 'H-H', 'S-S', 'C-F'}
    if bc_type not in allowed_bc_types:
        raise ValueError(f"bc_type must be one of {allowed_bc_types}, but got {bc_type}")

    allowed_distributions = {'X', 'U', 'O'}
    if distribution not in allowed_distributions:
        raise ValueError(f"distribution必须是{allowed_distributions}之一，但得到了{distribution}")

    allowed_optimizers = {'Adam', 'LBFGS', 'AdamW', 'RAdam', 'NAdam', 'Adamax'}
    if optimizer_type not in allowed_optimizers:
        raise ValueError(f"optimizer_type必须是{allowed_optimizers}之一，但得到了{optimizer_type}")

    allowed_activations = {'Tanh', 'Sin', 'SIREN'}
    if activation_type not in allowed_activations:
        raise ValueError(f"activation_type必须是{allowed_activations}之一，但得到了{activation_type}")

    if activation_type == 'SIREN':
        if siren_omega_0 <= 0:
            raise ValueError("siren_omega_0必须为正数")
        if siren_omega_hidden <= 0:
            raise ValueError("siren_omega_hidden必须为正数")

    allowed_lifting_basis = {'poly', 'polynomial', 'trig', 'sin', 'sincos', 'galerkin', 'none', 'identity', 'soft', 'raw'}
    lifting_basis_norm = str(lifting_basis).lower().strip()
    if lifting_basis_norm not in allowed_lifting_basis:
        raise ValueError(f"lifting_basis必须是{allowed_lifting_basis}之一，但得到了{lifting_basis}")

    allowed_network_archs = {'shared', 'encoder_decoder'}
    if network_arch not in allowed_network_archs:
        raise ValueError(f"network_arch必须是{allowed_network_archs}之一，但得到了{network_arch}")

    allowed_integrators = {'agq', 'gauss', 'clenshaw'}
    if integrator not in allowed_integrators:
        raise ValueError(f"integrator必须是{allowed_integrators}之一，但得到了{integrator}")

    allowed_samplers = {'uniform', 'latin_hypercube', 'sobol'}
    if sampler not in allowed_samplers:
        raise ValueError(f"sampler必须是{allowed_samplers}之一，但得到了{sampler}")

    return True

if __name__ != "__main__":
    try:
        validate_params()
    except Exception as e:
        print(f"[WARNING] Parameter validation failed: {e}")

def print_params_summary():
    print("\n" + "="*60)
    print("Parameter Configuration Summary")
    print("="*60)
    print(f"Geometry: h={h}m, L={L}m (L/h={L_factor}), layers={num_layers}")
    print(f"Material: W_Gr={W_Gr}, H_Gr={H_Gr}, T={T}K")
    print(f"Load: q={q}")
    print(f"Boundary: {bc_type}, Distribution: {distribution}")
    print(f"Elastic foundation: k1={k1}, k2={k2}")
    print(f"Initial defect: A0={A0}, a={a}, b={b}, c={c}")
    print(f"Training: epochs={epochs}, lr={lr}, N_train={N_train}")
    print(f"Network: {network_arch}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Integrator: {integrator} ({agq_rule})")
    print("="*60)

if __name__ == "__main__":
    print_params_summary()
    validate_params()
    print("[SUCCESS] Parameter validation passed!")
