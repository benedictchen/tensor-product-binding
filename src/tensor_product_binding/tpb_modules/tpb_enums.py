"""
🏷️ Tensor Product Binding - Enumerations Module  
===============================================

Split from tensor_product_binding.py (1103 lines → modular architecture)
Part of tensor_product_binding package 800-line compliance initiative.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Smolensky (1990) "Tensor Product Variable Binding and the Representation of Symbolic Structures"

🎯 MODULE PURPOSE:
=================
Core enumerations for tensor product binding operations.
Defines the types of mathematical operations available for binding vectors.

🔬 RESEARCH FOUNDATION:
======================
Based on Smolensky (1990) theoretical framework with modern extensions:
- OUTER_PRODUCT: Classic tensor product (role ⊗ filler)
- CIRCULAR_CONVOLUTION: Memory-efficient circular convolution  
- ADDITION/MULTIPLICATION: Simple operations for basic binding

This module contains the enumeration definitions, split from the
1103-line monolith for clean separation of concerns.
"""

from enum import Enum


class BindingOperation(Enum):
    """
    🔗 Types of binding operations available in tensor product binding.
    
    Different mathematical approaches to combine role and filler vectors:
    - OUTER_PRODUCT: Standard tensor product (role ⊗ filler)
    - CIRCULAR_CONVOLUTION: Circular convolution binding (memory efficient) 
    - ADDITION: Simple vector addition (least structured)
    - MULTIPLICATION: Element-wise multiplication (component binding)
    """
    OUTER_PRODUCT = "outer_product"
    CIRCULAR_CONVOLUTION = "circular_convolution"  
    ADDITION = "addition"
    MULTIPLICATION = "multiplication"


# Export the enumeration
__all__ = ['BindingOperation']


if __name__ == "__main__":
    print("🏷️ Tensor Product Binding - Enumerations Module")
    print("=" * 50)
    print("📊 MODULE CONTENTS:")
    print("  • BindingOperation - Core binding operation types")
    print("  • Research-accurate enumeration of TPB mathematical operations")
    print("")
    print("✅ Enumerations module loaded successfully!")
    print("🔬 Essential enums for tensor product binding operations!")