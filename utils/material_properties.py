
from typing import Tuple, Dict, NamedTuple, Optional
from dataclasses import dataclass
import numpy as np
import warnings

@dataclass
class MaterialConstants:
    T_0: float = 300.0
    
    E_Cu: float = 65.79e9
    nu_Cu: float = 0.387
    rho_Cu: float = 8.80e3
    alpha_Cu: float = 16.51e-6
    
    E_Gr: float = 929.57e9
    nu_Gr: float = 0.220
    rho_Gr: float = 1.80e3
    alpha_Gr: float = -3.98e-6
    
    l_Gr: float = 83.76
    t_Gr: float = 3.4

class MaterialProperties(NamedTuple):
    E: float
    nu: float
    alpha: float
    rho: float
    layer_properties: Dict
    Q11: Dict
    Q55: Dict

class DimensionlessParameters(NamedTuple):
    a11: float
    b11: float
    d11: float
    a55: float
    lambda_val: float
    n_xT: float
    m_xT: float

class MaterialCalculator:
    
    def __init__(self, constants: Optional[MaterialConstants] = None):
        self.constants = constants if constants is not None else MaterialConstants()
    
    def compute_correction_factors(self, V_Gr: float, T_ratio: float, H_Gr: float) -> Tuple[float, float, float, float]:
        if V_Gr == 0:
            return 1.0, 1.0, 1.0, 1.0
        
        V_Gr = np.clip(V_Gr, 0.0, 1.0)
        H_Gr = np.clip(H_Gr, 0.0, 10.0)
        T_ratio = np.clip(T_ratio, 0.5, 3.0)
        
        f_E = (1.11 - 1.22 * V_Gr - 0.134 * T_ratio + 0.559 * V_Gr * T_ratio
               - 5.5 * H_Gr * V_Gr + 38 * H_Gr * V_Gr ** 2 - 20.6 * H_Gr ** 2 * V_Gr ** 2)
        
        f_nu = (1.01 - 1.43 * V_Gr + 0.165 * T_ratio - 16.8 * H_Gr * V_Gr
                - 1.1 * H_Gr * V_Gr * T_ratio + 16 * H_Gr ** 2 * V_Gr ** 2)
        
        f_alpha = 0.794 - 16.8 * V_Gr ** 2 - 0.0279 * T_ratio * (1 + V_Gr)
        
        f_rho = 1.01 - 2.01 * V_Gr ** 2 - 0.0131 * T_ratio
        
        
        return f_E, f_nu, f_alpha, f_rho
    
    def compute_layer_properties(self, layer_index: int, total_layers: int, 
                               V_Gr_base: float, T_ratio: float, H_Gr: float,
                               distribution_type: str = 'X') -> Tuple[float, float, float, float]:
        if total_layers == 1:
            V_Gr_layer = V_Gr_base
        else:
            k = layer_index
            N = total_layers
            
            if distribution_type.upper() == 'X':
                V_Gr_layer = 2 * V_Gr_base * abs(2 * k - N - 1) / N
            elif distribution_type.upper() == 'O':
                V_Gr_layer = 2 * V_Gr_base * (1 - abs(2 * k - N - 1) / N)
            elif distribution_type.upper() == 'U':
                V_Gr_layer = V_Gr_base
            else:
                raise ValueError(f"不支持的分布类型: {distribution_type}")
        
        V_Gr_layer = np.clip(V_Gr_layer, 0.0, 1.0)
        V_Cu_layer = 1.0 - V_Gr_layer
        
        f_E, f_nu, f_alpha, f_rho = self.compute_correction_factors(V_Gr_layer, T_ratio, H_Gr)
        
        ksai = 2 * (self.constants.l_Gr / self.constants.t_Gr)
        eta = ((self.constants.E_Gr / self.constants.E_Cu - 1) / 
               (self.constants.E_Gr / self.constants.E_Cu + ksai))
        
        E = ((1 + ksai * eta * V_Gr_layer) / (1 - eta * V_Gr_layer) * 
             self.constants.E_Cu * f_E)
        
        nu = ((self.constants.nu_Gr * V_Gr_layer + 
               self.constants.nu_Cu * V_Cu_layer) * f_nu)
        
        alpha = ((self.constants.alpha_Gr * V_Gr_layer + 
                  self.constants.alpha_Cu * V_Cu_layer) * f_alpha)
        
        rho = ((self.constants.rho_Gr * V_Gr_layer + 
                self.constants.rho_Cu * V_Cu_layer) * f_rho)
        
        if E <= 0 or nu >= 1 or rho <= 0:
            warnings.warn(f"第{layer_index}层材料属性异常: E={E:.2e}, nu={nu:.3f}, rho={rho:.2e}")
        
        return E, nu, alpha, rho
    
    def compute_material_properties(self, h: float, num_layers: int, 
                                  W_Gr: float = 0.025, H_Gr: float = 0.8, 
                                  T: float = 300.0, distribution_type: str = 'X') -> MaterialProperties:
        T_ratio = T / self.constants.T_0
        
        if W_Gr > 0:
            V_Gr_base = W_Gr / (W_Gr + (self.constants.rho_Gr / self.constants.rho_Cu) * (1 - W_Gr))
        else:
            V_Gr_base = 0.0
        
        layer_props = {}
        Q11_layer = {}
        Q55_layer = {}
        
        E_effective = 0.0
        nu_effective = 0.0
        alpha_effective = 0.0
        rho_effective = 0.0
        
        layer_thickness = h / num_layers
        
        for k in range(1, num_layers + 1):
            E, nu, alpha, rho = self.compute_layer_properties(
                k, num_layers, V_Gr_base, T_ratio, H_Gr, distribution_type
            )
            
            if num_layers == 1:
                V_Gr_actual = V_Gr_base
            else:
                if distribution_type.upper() == 'X':
                    V_Gr_actual = 2 * V_Gr_base * abs(2*k - num_layers - 1) / num_layers
                elif distribution_type.upper() == 'O':
                    V_Gr_actual = 2 * V_Gr_base * (1 - abs(2*k - num_layers - 1) / num_layers)
                else:
                    V_Gr_actual = V_Gr_base
            
            layer_props[f"layer_{k}"] = {
                "E": E, "nu": nu, "alpha": alpha, "rho": rho,
                "V_Gr": V_Gr_actual
            }
            
            Q11 = E / (1 - nu ** 2)
            Q55 = E / (2 * (1 + nu))
            Q11_layer[f"Q11_{k}"] = Q11
            Q55_layer[f"Q55_{k}"] = Q55
            
            weight = layer_thickness / h
            E_effective += E * weight
            nu_effective += nu * weight
            alpha_effective += alpha * weight
            rho_effective += rho * weight
        
        return MaterialProperties(
            E=E_effective,
            nu=nu_effective, 
            alpha=alpha_effective,
            rho=rho_effective,
            layer_properties=layer_props,
            Q11=Q11_layer,
            Q55=Q55_layer
        )
    
    def compute_stiffness_coefficients(self, material_props: MaterialProperties, 
                                     h: float, num_layers: int) -> Tuple[float, float, float, float]:
        delta_z = h / num_layers
        
        A11 = 0.0
        B11 = 0.0
        D11 = 0.0
        A55 = 0.0
        
        Q11_values = list(material_props.Q11.values())
        Q55_values = list(material_props.Q55.values())
        
        kappa = 5.0 / 6.0
        
        for k in range(num_layers):
            z_k = -h / 2.0 + k * delta_z
            z_k_plus_1 = -h / 2.0 + (k + 1) * delta_z
            
            Q11_k = Q11_values[k]
            Q55_k = Q55_values[k]
            
            A11 += Q11_k * (z_k_plus_1 - z_k)
            
            B11 += Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0
            
            D11 += Q11_k * (z_k_plus_1**3 - z_k**3) / 3.0
            
            A55 += kappa * Q55_k * (z_k_plus_1 - z_k)
        
        return A11, B11, D11, A55
    
    def compute_thermal_forces(self, material_props: MaterialProperties, 
                             A11: float, B11: float, delta_T: float,
                             h: float, num_layers: int) -> Tuple[float, float]:
        Q11_values = list(material_props.Q11.values())
        
        A11_MN = 0.0
        B11_MN = 0.0
        N_XT = 0.0
        M_XT = 0.0
        
        delta_z = h / num_layers
        
        for k in range(num_layers):
            layer_key = f"layer_{k+1}"
            if layer_key in material_props.layer_properties:
                alpha_c_k = material_props.layer_properties[layer_key]['alpha']
            else:
                alpha_c_k = material_props.alpha
            
            Q11_k = Q11_values[k]
            
            z_k = -h/2.0 + k * delta_z
            z_k_plus_1 = -h/2.0 + (k + 1) * delta_z
            
            contribution_A11 = Q11_k * (z_k_plus_1 - z_k)
            A11_MN = A11_MN + contribution_A11
            N_XT = -A11_MN * alpha_c_k * delta_T
            
            contribution_B11 = Q11_k * (z_k_plus_1**2 - z_k**2) / 2.0
            B11_MN = B11_MN + contribution_B11
            M_XT = -B11_MN * alpha_c_k * delta_T
        
        return N_XT, M_XT
    
    def compute_dimensionless_parameters(self, h: float, L: float, num_layers: int,
                                       W_Gr: float = 0.025, H_Gr: float = 0.8,
                                       T: float = 300.0, distribution_type: str = 'X') -> DimensionlessParameters:
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)
        
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)
        
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)
        
        a11 = A11 / A11_0
        a55 = A55 / A11_0
        b11 = B11 / (h * A11_0)
        d11 = D11 / (h**2 * A11_0)
        
        lambda_val = L / h
        
        n_xT = N_XT / A11_0
        m_xT = M_XT / (h * A11_0)
        
        return DimensionlessParameters(
            a11=a11, b11=b11, d11=d11, a55=a55,
            lambda_val=lambda_val, n_xT=n_xT, m_xT=m_xT
        )
    
    def print_material_summary(self, h: float, L: float, num_layers: int,
                             W_Gr: float = 0.025, H_Gr: float = 0.8,
                             T: float = 300.0, distribution_type: str = 'X'):
        print("=" * 60)
        print("Material Properties Summary")
        print("=" * 60)
        print(f"Geometry: L = {L:.3f} m, h = {h:.3f} m, layers = {num_layers}")
        print(f"Material: W_Gr = {W_Gr:.3f}, H_Gr = {H_Gr:.3f}, T = {T:.1f} K")
        print(f"Distribution: {distribution_type}")
        print("-" * 60)
        
        props = self.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
        A11, B11, D11, A55 = self.compute_stiffness_coefficients(props, h, num_layers)
        
        props_0 = self.compute_material_properties(h, num_layers, 0.0, 0.0, T, distribution_type)
        A11_0, B11_0, D11_0, A55_0 = self.compute_stiffness_coefficients(props_0, h, num_layers)
        
        delta_T = T - self.constants.T_0
        N_XT, M_XT = self.compute_thermal_forces(props, A11, B11, delta_T, h, num_layers)
        
        print("Effective Material Properties:")
        print(f"  Young's modulus: E = {props.E/1e9:.2f} GPa")
        print(f"  Poisson's ratio: nu = {props.nu:.3f}")
        print(f"  Thermal expansion: alpha = {props.alpha*1e6:.2f} x 10^-6 /K")
        print(f"  Density: rho = {props.rho:.0f} kg/m^3")
        
        print("\nStiffness Coefficients:")
        print(f"  A11 = {A11:.2e}")
        print(f"  B11 = {B11:.2e}")
        print(f"  D11 = {D11:.2e}")
        print(f"  A55 = {A55:.2e}")

        print("\nPure Copper Reference:")
        print(f"  A11_0 = {A11_0:.2e}")
        print(f"  B11_0 = {B11_0:.2e}")
        print(f"  D11_0 = {D11_0:.2e}")
        print(f"  A55_0 = {A55_0:.2e}")

        dimensionless = self.compute_dimensionless_parameters(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)

        print("\nDimensionless Parameters:")
        print(f"  a11 = {dimensionless.a11:.4f}")
        print(f"  b11 = {dimensionless.b11:.4f}")
        print(f"  d11 = {dimensionless.d11:.4f}")
        print(f"  a55 = {dimensionless.a55:.4f}")
        print(f"  lambda = L/h = {dimensionless.lambda_val:.2f}")
        print(f"  n_xT = {dimensionless.n_xT:.6f}")
        print(f"  m_xT = {dimensionless.m_xT:.6f}")

        if delta_T != 0:
            print(f"\nThermal Parameters (dT = {delta_T:.1f} K):")
            print(f"  N_XT = {N_XT:.2e}")
            print(f"  M_XT = {M_XT:.2e}")
        
        print("=" * 60)

def create_material_calculator(constants: Optional[MaterialConstants] = None) -> MaterialCalculator:
    return MaterialCalculator(constants)

def compute_material_params_for_solver(h: float, L: float, num_layers: int = 10,
                                     W_Gr: float = 0.025, H_Gr: float = 0.8,
                                     T: float = 300.0, distribution_type: str = 'X',
                                     q: float = -0.08) -> Dict[str, float]:
    calculator = create_material_calculator()
    
    props = calculator.compute_material_properties(h, num_layers, W_Gr, H_Gr, T, distribution_type)
    dimensionless = calculator.compute_dimensionless_parameters(
        h, L, num_layers, W_Gr, H_Gr, T, distribution_type
    )
    
    delta_T = T - calculator.constants.T_0
    
    return {
        'a11': dimensionless.a11,
        'b11': dimensionless.b11,
        'd11': dimensionless.d11,
        'a55': dimensionless.a55,
        'lambda_val': dimensionless.lambda_val,
        'n_xT': dimensionless.n_xT,
        'm_xT': dimensionless.m_xT,
        'alpha_effective': props.alpha,
        'delta_T': delta_T,
        'q': q
    }

if __name__ == "__main__":
    print("Testing material properties calculation module...")
    
    calculator = create_material_calculator()
    
    h = 0.1
    L = 20 * h
    num_layers = 10
    W_Gr = 0.025
    H_Gr = 1
    T = 325.0
    q = -0.08
    distribution_type = 'X'
    calculator.print_material_summary(h, L, num_layers, W_Gr, H_Gr, T, distribution_type)
    
    solver_params = compute_material_params_for_solver(h, L, num_layers, W_Gr, H_Gr, T, distribution_type, q)
    
    print("\nSolver parameters:")
    for key, value in solver_params.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nMaterial properties module test completed!")
