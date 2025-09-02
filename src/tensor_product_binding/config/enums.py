"""
🏷️ Enumerations for Tensor Product Binding System
==================================================

This module defines all enumeration types used throughout the tensor product
binding system. These enums provide type-safe configuration options and
standardize the various methods and types used in TPB operations.

Based on:
- Smolensky (1990) tensor product variable binding theory
- Modern extensions for neural architectures
"""

from enum import Enum


class BindingOperation(Enum):
    """
    🔗 Types of binding operations for tensor product binding.
    
    Mathematical approaches to combine role and filler vectors:
    - OUTER_PRODUCT: Standard tensor product (role ⊗ filler) 
    - CIRCULAR_CONVOLUTION: Circular convolution binding (memory efficient)
    - ADDITION: Simple vector addition (least structured)
    - MULTIPLICATION: Element-wise multiplication (component binding)
    - HOLOGRAPHIC_REDUCED: HRR-style reduced representation
    """
    OUTER_PRODUCT = "outer_product"
    CIRCULAR_CONVOLUTION = "circular_convolution"
    ADDITION = "addition" 
    MULTIPLICATION = "multiplication"
    TENSOR_PRODUCT = "tensor_product"  # Legacy alias for OUTER_PRODUCT
    HOLOGRAPHIC_REDUCED = "holographic_reduced"
    VECTOR_MATRIX_MULTIPLICATION = "vector_matrix_multiplication"


class BindingMethod(Enum):
    """
    🧠 Methods for tensor product variable binding.
    
    Different strategies for performing the binding operation:
    - BASIC_OUTER_PRODUCT: Simple outer product R ⊗ F
    - RECURSIVE_BINDING: Hierarchical binding for nested structures 
    - CONTEXT_DEPENDENT: Context-sensitive binding for ambiguous roles
    - WEIGHTED_BINDING: Binding with strength modulation
    - MULTI_DIMENSIONAL: Different tensor dimensions per binding
    - HYBRID: Combine multiple methods
    """
    BASIC_OUTER_PRODUCT = "basic_outer"
    RECURSIVE_BINDING = "recursive"
    CONTEXT_DEPENDENT = "context_dependent"
    WEIGHTED_BINDING = "weighted"
    MULTI_DIMENSIONAL = "multi_dim"
    HYBRID = "hybrid"


class UnbindingMethod(Enum):
    """
    🔍 Methods for extracting information from tensor structures.
    
    Different approaches for unbinding fillers from bound representations:
    - BASIC_MULTIPLICATION: Simple matrix multiplication
    - LEAST_SQUARES: Optimal least-squares unbinding
    - REGULARIZED: Regularized unbinding for noise handling
    - ITERATIVE: Iterative unbinding for hierarchical structures
    - CONTEXT_SENSITIVE: Context-aware unbinding
    """
    BASIC_MULTIPLICATION = "basic_mult"
    LEAST_SQUARES = "least_squares"
    REGULARIZED = "regularized"
    ITERATIVE = "iterative"
    CONTEXT_SENSITIVE = "context_sensitive"


class VectorSpaceType(Enum):
    """
    🌌 Types of vector spaces in TPB system.
    
    Different semantic spaces for organizing vectors:
    - ROLE_SPACE: Space for role vectors (typically orthogonal)
    - FILLER_SPACE: Space for filler vectors (can be non-orthogonal)
    - BINDING_SPACE: Space for bound vectors (higher dimensional)
    - SUPERPOSITION_SPACE: Space for superposed representations
    """
    ROLE_SPACE = "role_space"
    FILLER_SPACE = "filler_space"
    BINDING_SPACE = "binding_space"
    SUPERPOSITION_SPACE = "superposition_space"


class StructureType(Enum):
    """
    🏗️ Types of structural representations.
    
    Different levels of structural complexity:
    - ATOMIC: Single role-filler binding
    - COMPOSITIONAL: Multiple bindings composed together
    - RECURSIVE: Hierarchical nested structures
    - SEQUENTIAL: Temporal binding sequences
    - CONTEXTUAL: Context-dependent bindings
    """
    ATOMIC = "atomic"
    COMPOSITIONAL = "compositional"
    RECURSIVE = "recursive" 
    SEQUENTIAL = "sequential"
    CONTEXTUAL = "contextual"


class OptimizationLevel(Enum):
    """
    ⚡ Optimization levels for performance tuning.
    
    Different performance vs accuracy trade-offs:
    - ACCURACY: Maximum accuracy, slower performance
    - BALANCED: Good balance of speed and accuracy
    - PERFORMANCE: Maximum speed, acceptable accuracy
    - MEMORY_EFFICIENT: Minimize memory usage
    """
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    MEMORY_EFFICIENT = "memory_efficient"


class NoiseModel(Enum):
    """
    🔊 Types of noise models for robust representations.
    
    Different approaches to handle noise and degradation:
    - GAUSSIAN: Additive Gaussian noise
    - UNIFORM: Uniform random noise
    - STRUCTURED: Systematic distortions
    - DROPOUT: Random vector component dropout
    """
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    STRUCTURED = "structured"
    DROPOUT = "dropout"