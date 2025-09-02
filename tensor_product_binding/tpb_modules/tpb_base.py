"""
🧠 Tensor Product Variable Binding - Revolutionary Neural-Symbolic Integration
==============================================================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to fully support continued research

Based on: Smolensky (1990) "Tensor Product Variable Binding and the Representation of Symbolic Structures"

🎯 ELI5 Summary:
Think of this like a super-smart filing system where you can store complex information 
(like "John loves Mary") by binding roles (subject, verb, object) with fillers (John, 
loves, Mary) using mathematical operations. It's like having structured sticky notes 
that never lose their organization, even when combined!

🔬 Research Background:
========================
Paul Smolensky's 1990 breakthrough solved the fundamental challenge of representing 
symbolic structure in neural networks. Before this, connectionist models could 
learn patterns but struggled with compositional structure like language syntax.

The TPR revolution:
- Systematic binding of variables (roles) with values (fillers)
- Compositional representations using tensor products
- Structured queries via unbinding operations
- Bridge between symbolic and connectionist AI
- Foundation for modern neural-symbolic reasoning

This launched the field of "neural symbolic integration" and influenced modern
architectures like Transformers and Graph Neural Networks.

🏗️ Architecture:
================
Role Vector (R) + Filler Vector (F) → Tensor Product (R ⊗ F)
Structure = Σ(R_i ⊗ F_i) for all role-filler pairs

🎨 ASCII Diagram - Tensor Product Binding:
=========================================
    Role Vector      Filler Vector      Tensor Product
    (Variable)         (Value)           (Binding)
    
    subject     ⊗     John         =     [2D Matrix]
      [r1]              [f1]              [r1×f1  r1×f2]
      [r2]              [f2]              [r2×f1  r2×f2]
      [r3]              [f3]              [r3×f1  r3×f2]
       ↓                 ↓                      ↓
   Structure Roles   Content Values     Bound Structure

Mathematical Framework:
- Binding: R_i ⊗ F_i (outer product)
- Structure: S = Σ(R_i ⊗ F_i) (superposition)
- Unbinding: F_j ≈ S @ R_j (approximate extraction)
- Query: Which filler is bound to role R_j?

🚀 Key Innovation: Distributed Structured Representations
Revolutionary Impact: Enables neural networks to process symbolic structures

⚡ Advanced Features:
====================
✨ Binding Methods:
  - basic_outer: Standard R ⊗ F binding
  - recursive: Hierarchical nested structures
  - context_dependent: Role disambiguation via context
  - weighted: Soft constraint binding strengths
  - multi_dimensional: Variable tensor dimensions
  - hybrid: Adaptive combination of methods

✨ Unbinding Methods:
  - basic_mult: Simple matrix multiplication
  - least_squares: Optimal overdetermined systems
  - regularized: Noise-robust with regularization
  - iterative: Complex hierarchical unbinding
  - context_sensitive: Context-aware extraction

✨ Structure Capabilities:
  - Compositional structure creation
  - Hierarchical representation
  - Structure similarity comparison
  - Multi-structure composition
  - Cleanup memory for robustness

Key Innovation: Mathematically rigorous method for representing symbolic structures
in distributed neural representations, enabling structured reasoning in connectionist systems!
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy.linalg import svd
from scipy.optimize import minimize
import sys
import os
warnings.filterwarnings('ignore')


class BindingOperation(Enum):
    """Different binding operation types"""
    TENSOR_PRODUCT = "tensor_product"
    CIRCULAR_CONVOLUTION = "circular_convolution"
    HOLOGRAPHIC_REDUCED = "holographic_reduced"
    VECTOR_MATRIX_MULTIPLICATION = "vector_matrix_multiplication"


class TPBVector:
    """Tensor Product Binding Vector with operations"""
    
    def __init__(self, data: np.ndarray):
        """Initialize TPB vector with data"""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data.astype(float)
    
    @property
    def dimension(self) -> int:
        """Get vector dimension"""
        return len(self.data)
    
    def magnitude(self) -> float:
        """Get vector magnitude/norm"""
        return np.linalg.norm(self.data)
    
    def normalize(self) -> 'TPBVector':
        """Return normalized copy of vector"""
        norm = self.magnitude()
        if norm > 0:
            return TPBVector(self.data / norm)
        return TPBVector(self.data.copy())
    
    def dot(self, other: 'TPBVector') -> float:
        """Compute dot product with another vector"""
        return np.dot(self.data, other.data)
    
    def cosine_similarity(self, other: 'TPBVector') -> float:
        """Compute cosine similarity with another vector"""
        norm1 = self.magnitude()
        norm2 = other.magnitude()
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = self.dot(other) / (norm1 * norm2)
        # Return absolute value for tensor product binding - orientation invariant
        return abs(similarity)
    
    def __add__(self, other: 'TPBVector') -> 'TPBVector':
        """Vector addition"""
        return TPBVector(self.data + other.data)
    
    def __sub__(self, other: 'TPBVector') -> 'TPBVector':
        """Vector subtraction"""
        return TPBVector(self.data - other.data)
    
    def __mul__(self, scalar: float) -> 'TPBVector':
        """Scalar multiplication"""
        return TPBVector(self.data * scalar)
    
    def __rmul__(self, scalar: float) -> 'TPBVector':
        """Right scalar multiplication"""
        return self.__mul__(scalar)
    
    def __repr__(self) -> str:
        return f"TPBVector({self.data})"

# Add parent directory to path for donation_utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from donation_utils import show_donation_message, show_completion_message


class BindingMethod(Enum):
    """Methods for tensor product variable binding"""
    BASIC_OUTER_PRODUCT = "basic_outer"      # Simple outer product R ⊗ F
    RECURSIVE_BINDING = "recursive"          # Hierarchical binding for nested structures 
    CONTEXT_DEPENDENT = "context_dependent"  # Context-sensitive binding for ambiguous roles
    WEIGHTED_BINDING = "weighted"            # Binding with strength modulation
    MULTI_DIMENSIONAL = "multi_dim"          # Different tensor dimensions per binding
    HYBRID = "hybrid"                        # Combine multiple methods


class UnbindingMethod(Enum):
    """Methods for extracting information from tensor structures"""
    BASIC_MULTIPLICATION = "basic_mult"      # Simple matrix multiplication
    LEAST_SQUARES = "least_squares"          # Optimal least-squares unbinding
    REGULARIZED = "regularized"              # Regularized unbinding for noise handling
    ITERATIVE = "iterative"                  # Iterative unbinding for hierarchical structures
    CONTEXT_SENSITIVE = "context_sensitive"  # Context-aware unbinding
    

@dataclass
class TensorBindingConfig:
    """Configuration for advanced tensor product binding with maximum flexibility"""
    
    # Core binding method
    binding_method: BindingMethod = BindingMethod.HYBRID
    
    # Binding strength and modulation
    enable_binding_strength: bool = True
    default_binding_strength: float = 1.0
    strength_decay_factor: float = 0.95  # For temporal binding sequences
    
    # Context-dependent binding settings
    context_window_size: int = 3
    context_sensitivity: float = 0.5
    enable_role_ambiguity_resolution: bool = True
    
    # Recursive/hierarchical binding settings
    max_recursion_depth: int = 5
    recursive_strength_decay: float = 0.8
    enable_hierarchical_unbinding: bool = True
    
    # Multi-dimensional tensor settings  
    enable_variable_dimensions: bool = False
    role_dimension_map: Optional[Dict[str, int]] = None
    filler_dimension_map: Optional[Dict[str, int]] = None
    
    # Unbinding configuration
    unbinding_method: UnbindingMethod = UnbindingMethod.REGULARIZED
    regularization_lambda: float = 0.001
    max_unbinding_iterations: int = 100
    unbinding_tolerance: float = 1e-6
    
    # Noise and robustness settings
    noise_tolerance: float = 0.1
    enable_cleanup_memory: bool = True
    cleanup_threshold: float = 0.7
    
    # Performance settings  
    enable_caching: bool = True
    enable_gpu_acceleration: bool = False  # For future GPU implementations


@dataclass
class BindingPair:
    """Represents a variable-value binding pair with advanced configuration"""
    variable: str
    value: Union[str, np.ndarray]
    role_vector: Optional[np.ndarray] = None
    filler_vector: Optional[np.ndarray] = None
    binding_strength: float = 1.0
    context: Optional[List[str]] = None
    hierarchical_level: int = 0


class TensorProductBinding:
    """
    Tensor Product Variable Binding System following Smolensky's original formulation
    
    The key insight: Use tensor products to bind variables (roles) with values (fillers)
    in a way that preserves both the structure and allows distributed processing.
    
    Mathematical foundation:
    - Role vectors R_i represent variables/positions
    - Filler vectors F_i represent values/content  
    - Binding: R_i ⊗ F_i (tensor product)
    - Complex structure: Σ_i R_i ⊗ F_i
    """
    
    def __init__(
        self,
        vector_dim: int = 100,
        symbol_dim: Optional[int] = None,
        role_dim: Optional[int] = None,
        role_vectors: Optional[Dict[str, np.ndarray]] = None,
        filler_vectors: Optional[Dict[str, np.ndarray]] = None,
        random_seed: Optional[int] = None,
        config: Optional[TensorBindingConfig] = None
    ):
        """
        🏗️ Initialize Tensor Product Variable Binding System
        
        🎯 ELI5: Think of this as setting up a super-smart organizational system where 
        you can store complex structured information (like sentences, databases, or 
        spatial relationships) by pairing "slots" with "values" using mathematical 
        magic that keeps everything organized even when mixed together!
        
        Technical Details:
        Initialize a TPR system implementing Smolensky's (1990) tensor product approach
        to representing symbolic structures in distributed neural representations.
        Enables systematic binding of structural roles with content fillers.
        
        Args:
            vector_dim (int): Dimension of role and filler vectors - the "size of slots/values"
                             Typical values: 50-500 (higher = more capacity, slower operations)
                             Must be same for roles and fillers for binding compatibility
            role_vectors (Optional[Dict[str, np.ndarray]]): Pre-defined role vectors (variables)
                                                          Dict mapping names to vectors
                                                          None = create new roles as needed
                                                          Use when you have fixed structure roles
            filler_vectors (Optional[Dict[str, np.ndarray]]): Pre-defined filler vectors (values)  
                                                            Dict mapping names to vectors
                                                            None = create new fillers as needed
                                                            Use when you have fixed content items
            random_seed (Optional[int]): Random seed for reproducibility
                                       Use same seed to get identical role/filler vectors
                                       Important for experiments and comparisons
            config (Optional[TensorBindingConfig]): Advanced configuration options
                                                  None = use default hybrid configuration
