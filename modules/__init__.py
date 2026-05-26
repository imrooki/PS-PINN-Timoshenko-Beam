
try:

    from .data_types import (
        MaterialCoeffs,
        PhysicalParams,
        BoundaryConditions,
        BoundaryConditionType,
        DistributionType,
    )

    from .numerics import d_dx, compute_derivatives, safe_divide

    from .nets import (
        SharedEncoderMultiDecoder,
        build_timoshenko_net,
    )

    from .bc import BoundaryConditionPenalty, lifting

    from .physics import (
        EnergyLoss,
        WeightedEnergyLoss,
        create_loss_function,
    )

    from .solver import EnergyPINNStatic, as_fun, train_model

except ImportError:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    from modules.data_types import (
        MaterialCoeffs,
        PhysicalParams,
        BoundaryConditions,
        BoundaryConditionType,
        DistributionType,
    )
    from modules.numerics import d_dx, compute_derivatives, safe_divide
    from modules.nets import (
        SharedEncoderMultiDecoder,
        build_timoshenko_net,
    )
    from modules.bc import BoundaryConditionPenalty, lifting
    from modules.physics import (
        EnergyLoss,
        WeightedEnergyLoss,
        create_loss_function,
    )
    from modules.solver import EnergyPINNStatic, as_fun, train_model

__all__ = [
    "MaterialCoeffs",
    "PhysicalParams",
    "BoundaryConditions",
    "BoundaryConditionType",
    "DistributionType",
    "SharedEncoderMultiDecoder",
    "build_timoshenko_net",
    "EnergyPINNStatic",
    "as_fun",
    "train_model",
    "EnergyLoss",
    "WeightedEnergyLoss",
    "create_loss_function",
    "BoundaryConditionPenalty",
    "lifting",
    "d_dx",
    "compute_derivatives",
    "safe_divide",
]
