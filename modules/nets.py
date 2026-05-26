
from __future__ import annotations

from typing import Optional, Type
import math
import torch
import torch.nn as nn

class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class SIREN(nn.Module):
    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)

def _siren_init_weights(m: nn.Module, omega_0: float = 30.0, is_first: bool = False) -> None:
    if isinstance(m, nn.Linear):
        in_features = m.weight.shape[1]
        if is_first:
            bound = 1.0 / in_features
        else:
            bound = math.sqrt(6.0 / in_features) / omega_0
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_activation_class(activation_type: str, **kwargs) -> Type[nn.Module]:
    activation_map = {
        'Tanh': nn.Tanh,
        'Sin': Sin,
        'SIREN': lambda: SIREN(omega=kwargs.get('omega', 30.0)),
    }

    act_type = activation_type.strip()
    if act_type not in activation_map:
        raise ValueError(
            f"不支持的激活函数类型: {act_type}. "
            f"可选项: {list(activation_map.keys())}"
        )

    return activation_map[act_type]

class SharedEncoderMultiDecoder(nn.Module):

    def __init__(
        self,
        in_dim: int = 2,
        activation: Type[nn.Module] = nn.Tanh,
        encoder_dims: Optional[list] = None,
        head_dims: Optional[list] = None,
        use_siren_init: bool = False,
        siren_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if encoder_dims is None or head_dims is None:
            raise ValueError("SharedEncoderMultiDecoder requires encoder_dims and head_dims")

        self.use_siren_init = use_siren_init
        self.siren_omega_0 = siren_omega_0

        enc_layers = []
        for i in range(len(encoder_dims) - 1):
            enc_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            if callable(activation):
                try:
                    act_instance = activation()
                except TypeError:
                    act_instance = activation
            else:
                act_instance = activation
            enc_layers.append(act_instance if isinstance(act_instance, nn.Module) else activation())
        self.encoder = nn.Sequential(*enc_layers)

        self.head_u = self._build_head(head_dims, activation)
        self.head_w = self._build_head(head_dims, activation)
        self.head_phi = self._build_head(head_dims, activation)

        self.encoder_dims = encoder_dims
        self.head_dims = head_dims

        if use_siren_init:
            self._apply_siren_init()
        else:
            self.apply(self._init_weights)

    def _apply_siren_init(self) -> None:
        first_layer = True
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                _siren_init_weights(module, self.siren_omega_0, is_first=first_layer)
                first_layer = False

        for head in [self.head_u, self.head_w, self.head_phi]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    _siren_init_weights(module, self.siren_omega_0, is_first=False)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _build_head(head_dims: list, activation) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(len(head_dims) - 1):
            layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            if i < len(head_dims) - 2:
                if callable(activation):
                    try:
                        act_instance = activation()
                    except TypeError:
                        act_instance = activation
                else:
                    act_instance = activation
                layers.append(act_instance if isinstance(act_instance, nn.Module) else activation())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        u = self.head_u(z)
        w = self.head_w(z)
        phi = self.head_phi(z)
        return torch.cat([u, w, phi], dim=1)

def build_timoshenko_net(
    *,
    in_dim: int = 1,
    activation: Optional[Type[nn.Module]] = None,
    activation_type: Optional[str] = None,
    encoder_dims_shared: Optional[list] = None,
    head_dims: Optional[list] = None,
    siren_omega_0: float = 30.0,
    siren_omega_hidden: float = 30.0,
    **kwargs,
) -> nn.Module:

    if encoder_dims_shared is None or head_dims is None:
        raise ValueError("SharedEncoder requires 'encoder_dims_shared' and 'head_dims'")
    if not isinstance(in_dim, int) or in_dim <= 0:
        raise ValueError(f"in_dim must be a positive int, got: {in_dim}")
    if encoder_dims_shared[0] != in_dim:
        raise ValueError(
            f"encoder_dims_shared[0] ({encoder_dims_shared[0]}) must match in_dim ({in_dim})"
        )
    if head_dims[-1] != 1:
        raise ValueError("head_dims must end with 1 (scalar head output)")

    use_siren_init = False
    if activation_type is not None:
        act_type = activation_type.strip()
        if act_type == 'SIREN':
            activation = lambda: SIREN(omega=siren_omega_hidden)
            use_siren_init = True
        elif act_type == 'Sin':
            activation = Sin
        elif act_type == 'Tanh':
            activation = nn.Tanh
        else:
            raise ValueError(f"不支持的 activation_type: {act_type}. 可选: ['Tanh', 'Sin', 'SIREN']")
    elif activation is None:
        activation = nn.Tanh

    return SharedEncoderMultiDecoder(
        in_dim=in_dim,
        activation=activation,
        encoder_dims=encoder_dims_shared,
        head_dims=head_dims,
        use_siren_init=use_siren_init,
        siren_omega_0=siren_omega_0,
    )

__all__ = [
    "Sin",
    "SIREN",
    "get_activation_class",
    "SharedEncoderMultiDecoder",
    "build_timoshenko_net",
]
