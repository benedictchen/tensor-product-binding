"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
❤️ https://github.com/sponsors/benedictchen

Your support makes advanced AI research accessible to everyone! 🚀

Tensor Product Variable Binding Library
=======================================

Based on: Smolensky (1990) "Tensor Product Variable Binding and the Representation of Symbolic Structures"

This library implements the foundational method for representing structured knowledge 
in neural networks using tensor products to bind variables with values.

🔬 Research Foundation:
- Paul Smolensky's Tensor Product Variable Binding
- Tony Plate's Holographic Reduced Representations (HRR) 
- Vector Symbolic Architecture (VSA) principles
- Distributed representation of symbolic structures

🎯 Key Features:
- Tensor product binding operations
- Vector symbolic representation
- Compositional semantics
- Neural binding networks
- Modular architecture for flexibility
"""

from .symbolic_structures import SymbolicStructureEncoder as _SSE, TreeNode as _TreeNode
from .neural_binding import NeuralBindingNetwork as _NBN
from .compositional_semantics import CompositionalSemantics as _CS

class SymbolicStructureEncoder:
    def __init__(self, **kwargs):
        self._impl = _SSE(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self._impl, name)

class TreeNode:
    def __init__(self, **kwargs):
        self._impl = _TreeNode(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self._impl, name)

class NeuralBindingNetwork:
    def __init__(self, **kwargs):
        self._impl = _NBN(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self._impl, name)

class CompositionalSemantics:
    def __init__(self, **kwargs):
        self._impl = _CS(**kwargs)
    
    def __getattr__(self, name):
        return getattr(self._impl, name)

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        print("\n🧮 Tensor Product Binding Library - Made possible by Benedict Chen")
        print("   \033]8;;mailto:benedict@benedictchen.com\033\\benedict@benedictchen.com\033]8;;\033\\")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   🔗 \033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\033\\💳 CLICK HERE TO DONATE VIA PAYPAL\033]8;;\033\\")
        print("   ❤️ \033]8;;https://github.com/sponsors/benedictchen\033\\💖 SPONSOR ON GITHUB\033]8;;\033\\")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")
        print("")
    except:
        print("\n🧮 Tensor Product Binding Library - Made possible by Benedict Chen")
        print("   benedict@benedictchen.com")
        print("")
        print("💰 PLEASE DONATE! Your support keeps this research alive! 💰")
        print("   💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")
        print("   ❤️ GitHub: https://github.com/sponsors/benedictchen")
        print("")
        print("   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!")
        print("   (Start small, dream big! Every donation helps! 😄)")

from .tpb_modules import (
    BindingOperation,
    BindingMethod, 
    UnbindingMethod,
    TensorBindingConfig,
    BindingPair,
    TPBVector,
    CoreBinding
)

from .tensor_product_binding import TensorProductBinding as _TPB

class TensorProductBinding:
    def __init__(self, vector_dim: int = 100, role_dimension: int = None, 
                 filler_dimension: int = None, **kwargs):
        if role_dimension is None:
            role_dimension = vector_dim
        if filler_dimension is None:
            filler_dimension = vector_dim
            
        self._impl = _TPB(
            role_dimension=role_dimension,
            filler_dimension=filler_dimension,
            **kwargs
        )
    
    def __getattr__(self, name):
        return getattr(self._impl, name)

def create_neural_binding_network(network_type="pytorch", *args, **kwargs):
    from .neural_binding import PyTorchBindingNetwork, NumPyBindingNetwork
    
    if network_type.lower() == "pytorch":
        return PyTorchBindingNetwork(*args, **kwargs)
    elif network_type.lower() == "numpy":
        return NumPyBindingNetwork(*args, **kwargs)
    else:
        return NeuralBindingNetwork(*args, **kwargs)

def create_tpb_system(*args, **kwargs):
    return TensorProductBinding(*args, **kwargs)

# Show attribution on library import
_print_attribution()

__version__ = "1.0.0"
__authors__ = ["Based on Smolensky (1990)"]

__all__ = [
    # Core modular classes (implemented)
    "BindingOperation",
    "BindingMethod", 
    "UnbindingMethod",
    "TensorBindingConfig", 
    "BindingPair",
    "TPBVector",
    "CoreBinding",
    
    # Main interface (backward compatibility)
    "TensorProductBinding",
    
    # Connected scattered implementations (REAL)
    "SymbolicStructureEncoder", 
    "TreeNode",
    "NeuralBindingNetwork",
    "CompositionalSemantics",
    
    # Placeholders that still need implementation
    "SymbolicStructure",
    "StructureType", 
    "PyTorchBindingNetwork",
    "NumPyBindingNetwork", 
    "create_neural_binding_network",
    "ConceptualSpace",
    "SemanticRole"
]

"""
💝 Thank you for using this research software! 💝

📚 If this work contributed to your research, please:
💳 DONATE: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
❤️ SPONSOR: https://github.com/sponsors/benedictchen
📝 CITE: Benedict Chen (2025) - Tensor Product Binding Research Implementation

Your support enables continued development of cutting-edge AI research tools! 🎓✨
"""