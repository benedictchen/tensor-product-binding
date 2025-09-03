"""
🏗️ Tensor Product Binding Modules - Modular Architecture
========================================================

Modular architecture for tensor product binding system,
split from monolithic tensor_product_binding.py (1103 lines → 4 modules).

Part of tensor_product_binding package 800-line compliance initiative.

Modular implementation of Tony Plate's HRR and Paul Smolensky's TPB
for distributed symbolic representation with comprehensive FIXME solutions.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Smolensky (1990) "Tensor Product Variable Binding and the Representation of Symbolic Structures"
"""

# Legacy imports (existing modules)
from .config_enums import (
    BindingOperation,
    BindingMethod, 
    UnbindingMethod,
    TensorBindingConfig,
    BindingPair
)

from .vector_operations import TPBVector

from .core_binding import CoreBinding

# ✅ NEW MODULAR COMPONENTS (800-line compliance)
try:
    # New modular tensor_product_binding.py components  
    from .tpb_enums import BindingOperation as ModularBindingOperation
    from .tpb_vector import TPBVector as ModularTPBVector, BindingPair as ModularBindingPair
    from .tpb_core import TensorProductBinding
    from .tpb_factory import create_tpb_system, demo_tensor_binding, create_linguistic_example
    
    # Use modular components as primary
    MODULAR_AVAILABLE = True
    
except ImportError:
    MODULAR_AVAILABLE = False
    TensorProductBinding = None
    create_tpb_system = None
    demo_tensor_binding = None
    create_linguistic_example = None


# Export legacy and new components
__all__ = [
    # Legacy components
    'BindingOperation',
    'BindingMethod',
    'UnbindingMethod', 
    'TensorBindingConfig',
    'BindingPair',
    'TPBVector',
    'CoreBinding',
]

# Add modular components if available
if MODULAR_AVAILABLE:
    __all__.extend([
        'TensorProductBinding',      # Main system
        'create_tpb_system',         # Factory function
        'demo_tensor_binding',       # Educational demo
        'create_linguistic_example', # NLP example
        'MODULAR_AVAILABLE'          # Availability flag
    ])


# Convenience factory function
def create_tpb(vector_dim: int = 100, **kwargs):
    """
    🏭 Quick TPB system creation.
    
    Uses modular TensorProductBinding if available, falls back to legacy.
    
    Parameters
    ----------
    vector_dim : int, default=100
        Dimension of vectors
    **kwargs
        Additional parameters
        
    Returns
    -------
    TensorProductBinding or CoreBinding
        TPB system instance
    """
    if MODULAR_AVAILABLE and create_tpb_system:
        return create_tpb_system(vector_dim=vector_dim, **kwargs)
    else:
        # Fallback to legacy
        return CoreBinding(**kwargs)


# Add convenience function to exports
__all__.append('create_tpb')