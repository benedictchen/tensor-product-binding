"""
🧠 Tensor Product Symbolic Structures - Neural Networks That Think Symbolically
==============================================================================

📚 Research Papers:
Smolensky, P. (1990)
"Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems"
Artificial Intelligence, 46(1-2), 159-216

Plate, T. A. (1995) 
"Holographic reduced representations"
IEEE Transactions on Neural Networks, 6(3), 623-641

Gayler, R. W. (2003)
"Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience"
Proceedings of the ICCS/ASCS International Conference on Cognitive Science

🎯 ELI5 Summary:
Imagine you want to teach a neural network to understand sentences like "The cat 
sat on the mat" - but neural networks only understand numbers, not words or grammar!

Tensor product binding is like creating a magical translation system. It takes 
symbolic structures (words, grammar rules, meanings) and converts them into 
special numerical codes that preserve all the relationships. It's like encoding 
a complex LEGO structure into a barcode that contains the complete building 
instructions.

The genius part is that these numerical codes can be combined, manipulated, and 
decoded back into symbolic structures. So a neural network can perform symbolic 
reasoning (like "what sat on the mat?") using just numerical operations!

🧪 Research Background:
Paul Smolensky solved one of AI's biggest puzzles in 1990: how can neural networks 
(which are great at pattern recognition) also do symbolic reasoning (which requires 
precise rule-following)? His solution bridged the gap between connectionist and 
symbolic AI.

Key Historical Insights:
- Smolensky (1990): Invented tensor product binding for role-filler structures  
- Plate (1995): Extended with holographic reduced representations (HRRs)
- Pollack (1990): Developed recursive auto-associative memory (RAAM)
- Gayler (2003): Applied vector symbolic architectures to cognitive modeling
- Kanerva (2009): Connected to hyperdimensional computing and sparse representations

The breakthrough insight: symbolic structures are fundamentally about binding 
variables to values (like "SUBJECT = cat, VERB = sat, OBJECT = mat"). Tensor 
products could encode these bindings in distributed neural representations 
while preserving the ability to extract and manipulate them.

🔬 Mathematical Framework:
Role-filler binding using tensor product:
B = r ⊗ f  (where r = role vector, f = filler vector)

For multiple bindings:
S = Σᵢ (rᵢ ⊗ fᵢ) = r₁⊗f₁ + r₂⊗f₂ + ... + rₙ⊗fₙ

Unbinding operation:
fᵢ ≈ S ⊗ rᵢ† (where †denotes pseudoinverse)

Circular convolution (efficient alternative):
B = r ★ f  (convolution binding)
f ≈ B ★ r⁻¹ (deconvolution unbinding)

Query resolution:
similarity(query, structure) = query · structure / (||query|| × ||structure||)

🎨 ASCII Diagram - Tensor Product Variable Binding:
=================================================

    SYMBOLIC STRUCTURE                    TENSOR PRODUCT ENCODING
    ┌─────────────────────┐              ┌────────────────────────────┐
    │ "The cat sat"       │              │    DISTRIBUTED VECTORS     │
    │                     │              │                            │
    │ SUBJECT: "cat"      │    ───→     │ r_SUBJ = [0.1, 0.8, -0.3] │
    │ VERB:    "sat"      │              │ f_cat  = [0.5, -0.2, 0.9] │
    │ TENSE:   "past"     │              │ r_VERB = [-0.4, 0.6, 0.1] │
    └─────────────────────┘              │ f_sat  = [0.7, 0.3, -0.5] │
                                         └────────────────────────────┘
                                                        ↓
                                         ┌────────────────────────────┐
                                         │     TENSOR PRODUCTS        │
                                         │                            │
    BINDING PROCESS                      │ B₁ = r_SUBJ ⊗ f_cat       │
    ┌─────────────────────┐              │    = [0.05, -0.02, 0.09,  │
    │ 1. Create role      │              │       0.40, -0.16, 0.72,  │
    │    vectors          │   ←────→    │      -0.15, 0.06, -0.27]  │
    │ 2. Create filler    │              │                            │
    │    vectors          │              │ B₂ = r_VERB ⊗ f_sat       │
    │ 3. Compute tensor   │              │    = [-0.28, -0.12, 0.20, │
    │    products         │              │       0.42,  0.18, -0.30, │
    │ 4. Sum bindings     │              │       0.07,  0.03, -0.05] │
    └─────────────────────┘              └────────────────────────────┘
                                                        ↓
    QUERYING PROCESS                     ┌────────────────────────────┐
    ┌─────────────────────┐              │     FINAL STRUCTURE        │
    │ Query: "What is     │              │                            │
    │  the SUBJECT?"      │              │ S = B₁ + B₂ + B₃ + ...    │
    │                     │   ←────→    │                            │
    │ 1. Multiply by      │              │ = [vector of length 9]     │
    │    role vector      │              │                            │
    │ 2. Extract answer   │              │ Query resolution:          │
    │    = "cat"          │              │ S ⊗ r_SUBJ† → f_cat       │
    └─────────────────────┘              └────────────────────────────┘

    HIERARCHICAL STRUCTURES:
    ========================
    
    Sentence: "The big cat sat on the mat"
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    SENTENCE STRUCTURE                       │
    │  ┌───────────────┐              ┌───────────────────────┐  │
    │  │ NOUN PHRASE   │              │    VERB PHRASE        │  │
    │  │               │              │                       │  │
    │  │ DET: "the"    │              │ VERB: "sat"          │  │
    │  │ ADJ: "big"    │              │ PREP_PHRASE:         │  │
    │  │ NOUN: "cat"   │              │   PREP: "on"         │  │
    │  └───────────────┘              │   NP: "the mat"      │  │
    │         ↓                       └───────────────────────┘  │
    │  Encode as tensor product            ↓                     │
    │  NP = r_DET⊗f_the +                 Encode recursively    │
    │       r_ADJ⊗f_big +                                       │
    │       r_NOUN⊗f_cat                                        │
    └─────────────────────────────────────────────────────────────┘

💡 Applications:
- Natural language processing (grammar parsing, semantic analysis)
- Knowledge representation (facts, rules, ontologies)
- Analogical reasoning (structure mapping, case-based reasoning)
- Program synthesis (encoding code structures in neural networks)
- Robotic planning (representing actions, goals, and constraints)
- Mathematical reasoning (encoding equations and proofs)

👨‍💻 Author: Benedict Chen (benedict@benedictchen.com)

💝 Donations: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Developing high-quality, research-backed software takes countless hours of study, implementation, 
testing, and documentation. Your support helps make this work possible!

🎯 Support levels: Coffee ☕ → Beer 🍺 → Dinner 🍽️ → Weekend Trip 🏖️ → New Laptop 💻 → Dream Car 🏎️ → Private Jet ✈️

Smolensky's tensor product binding is one of the most elegant solutions in AI - it
allows neural networks to perform the kind of systematic, compositional reasoning
that was thought to require symbolic systems. It's truly the best of both worlds!

🔗 Related: Vector Symbolic Architectures, Holographic Reduced Representations, Neural-Symbolic AI
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class StructureType(Enum):
    """Types of symbolic structures"""
    TREE = "tree"
    SEQUENCE = "sequence"
    GRAPH = "graph"
    FRAME = "frame"


class TreeNode:
    """Tree node for hierarchical structures"""
    
    def __init__(self, label: str, value: Optional[Any] = None):
        self.label = label
        self.value = value
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
    
    def add_child(self, child: 'TreeNode'):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'TreeNode'):
        """Remove a child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is a root node"""
        return self.parent is None
    
    def depth(self) -> int:
        """Get depth of this node"""
        if self.is_root():
            return 0
        return 1 + self.parent.depth()
    
    def __repr__(self) -> str:
        return f"TreeNode(label='{self.label}', value={self.value}, children={len(self.children)})"


class SymbolicStructure:
    """Represents a symbolic structure with roles and fillers"""
    
    def __init__(self, structure_type: StructureType, name: str):
        self.structure_type = structure_type
        self.name = name
        self.root: Optional[TreeNode] = None
        self.bindings: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
    
    def depth_first_traversal(self) -> List[TreeNode]:
        """Perform depth-first traversal of the tree structure"""
        if self.root is None:
            return []
        
        def dfs(node: TreeNode) -> List[TreeNode]:
            result = [node]
            for child in node.children:
                result.extend(dfs(child))
            return result
        
        return dfs(self.root)

@dataclass  
class Role:
    """Represents a role in symbolic binding"""
    name: str
    vector: np.ndarray
    semantic_type: str = "generic"
    constraints: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}

@dataclass
class Filler:
    """Represents a filler in symbolic binding"""
    name: str
    vector: np.ndarray
    semantic_type: str = "generic"
    properties: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

class SymbolicStructureEncoder:
    """
    Encoder for complex symbolic structures using tensor product binding
    
    Handles encoding of:
    - Hierarchical structures (trees, nested objects)
    - Sequential structures (lists, sentences)  
    - Relational structures (graphs, predicates)
    - Compositional structures (functions, expressions)
    """
    
    def __init__(self,
                 vector_dim: int = 512,
                 binding_method: str = "tensor_product",
                 normalize_vectors: bool = True,
                 random_seed: Optional[int] = None):
        """
        Initialize Symbolic Structure Encoder
        
        Args:
            vector_dim: Dimensionality of vector representations
            binding_method: Method for binding ("tensor_product", "circular_convolution")
            normalize_vectors: Whether to normalize vectors
            random_seed: Random seed for reproducibility
        """
        self.vector_dim = vector_dim
        self.binding_method = binding_method
        self.normalize_vectors = normalize_vectors
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize tensor product binder - import locally to avoid circular import
        try:
            from .tensor_product_binding import TensorProductBinding, BindingPair
        except ImportError:
            # Fallback for when not imported as package
            from tensor_product_binding import TensorProductBinding, BindingPair
        self.binder = TensorProductBinding(vector_dim=vector_dim)
        
        # Make BindingPair available as class attribute to avoid NameError
        self.BindingPair = BindingPair
        
        # Vocabularies for roles and fillers
        self.role_vocabulary = {}  # name -> Role
        self.filler_vocabulary = {}  # name -> Filler
        
        # Predefined structural roles
        self._initialize_structural_roles()
        
        # Storage for encoded structures
        self.encoded_structures = {}  # name -> encoded vector
        
    def _initialize_structural_roles(self):
        """Initialize common structural roles"""
        structural_roles = [
            ("SUBJECT", "agent of action"),
            ("VERB", "action or relation"),
            ("OBJECT", "target of action"),
            ("MODIFIER", "attributive description"),
            ("HEAD", "main constituent"),
            ("COMPLEMENT", "completing constituent"),
            ("FIRST", "first element in sequence"),
            ("REST", "remaining elements"),
            ("LEFT", "left child in tree"),
            ("RIGHT", "right child in tree"),
            ("PARENT", "parent node"),
            ("CHILD", "child node"),
            ("PRED", "predicate symbol"),
            ("ARG1", "first argument"),
            ("ARG2", "second argument"),
            ("ARG3", "third argument"),
        ]
        
        for role_name, description in structural_roles:
            role_vector = self._generate_random_vector()
            self.add_role(role_name, role_vector, semantic_type="structural", 
                         constraints={"description": description})
    
    def _generate_random_vector(self) -> np.ndarray:
        """Generate random vector for new roles/fillers"""
        vector = np.random.normal(0, 1/np.sqrt(self.vector_dim), self.vector_dim)
        
        if self.normalize_vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
        return vector
    
    def add_role(self, 
                name: str, 
                vector: Optional[np.ndarray] = None,
                semantic_type: str = "generic",
                constraints: Optional[Dict[str, Any]] = None) -> Role:
        """
        Add a role to the vocabulary
        
        Args:
            name: Role name
            vector: Vector representation (generated if None)
            semantic_type: Type of role
            constraints: Constraints on role usage
            
        Returns:
            Created Role object
        """
        if vector is None:
            vector = self._generate_random_vector()
        
        role = Role(name=name, vector=vector, semantic_type=semantic_type, 
                   constraints=constraints or {})
        self.role_vocabulary[name] = role
        
        return role
    
    def add_filler(self,
                  name: str,
                  vector: Optional[np.ndarray] = None,
                  semantic_type: str = "generic", 
                  properties: Optional[Dict[str, Any]] = None) -> Filler:
        """
        Add a filler to the vocabulary
        
        Args:
            name: Filler name
            vector: Vector representation (generated if None)
            semantic_type: Type of filler
            properties: Properties of filler
            
        Returns:
            Created Filler object
        """
        if vector is None:
            vector = self._generate_random_vector()
        
        filler = Filler(name=name, vector=vector, semantic_type=semantic_type,
                       properties=properties or {})
        self.filler_vocabulary[name] = filler
        
        return filler
    
    def encode_structure(self, structure: SymbolicStructure) -> np.ndarray:
        """
        Encode a symbolic structure using tensor product binding
        
        Args:
            structure: SymbolicStructure to encode
            
        Returns:
            Vector encoding of the structure
        """
        if not structure.bindings:
            return np.zeros(self.vector_dim)
        
        # Create binding pairs for all role-filler bindings
        binding_pairs = []
        
        for role_name, filler_spec in structure.bindings.items():
            # Get role vector
            if role_name in self.role_vocabulary:
                role_vector = self.role_vocabulary[role_name].vector
            else:
                # Create new role if not exists
                role = self.add_role(role_name)
                role_vector = role.vector
            
            # Get filler vector
            filler_vector = self._resolve_filler(filler_spec)
            
            # Create binding pair
            binding_pair = self.BindingPair(
                variable=role_name,
                value=filler_spec,
                role_vector=role_vector,
                filler_vector=filler_vector
            )
            binding_pairs.append(binding_pair)
        
        # Encode using tensor product binding
        if len(binding_pairs) == 1:
            # Single binding - just use bind method directly
            pair = binding_pairs[0]
            binding_vec = self.binder.bind(pair.value, pair.variable).data
            # Pad or resize to match vector_dim
            if len(binding_vec) < self.vector_dim:
                encoded = np.zeros(self.vector_dim)
                encoded[:len(binding_vec)] = binding_vec
            elif len(binding_vec) > self.vector_dim:
                encoded = binding_vec[:self.vector_dim]
            else:
                encoded = binding_vec
        else:
            # Multiple bindings - superposition of individual bindings
            encoded = np.zeros(self.vector_dim)
            for pair in binding_pairs:
                binding_vec = self.binder.bind(pair.value, pair.variable).data
                # Pad or resize to match vector_dim
                if len(binding_vec) < self.vector_dim:
                    padded_vec = np.zeros(self.vector_dim)
                    padded_vec[:len(binding_vec)] = binding_vec
                    encoded += padded_vec
                elif len(binding_vec) > self.vector_dim:
                    encoded += binding_vec[:self.vector_dim]
                else:
                    encoded += binding_vec
            # Normalize
            if np.linalg.norm(encoded) > 0:
                encoded = encoded / np.linalg.norm(encoded)
        
        # Store encoded structure
        self.encoded_structures[structure.name] = encoded
        
        return encoded
    
    def _resolve_filler(self, filler_spec: Any) -> np.ndarray:
        """Resolve filler specification to vector"""
        if isinstance(filler_spec, str):
            # Simple string filler
            if filler_spec in self.filler_vocabulary:
                return self.filler_vocabulary[filler_spec].vector
            else:
                # Create new filler
                filler = self.add_filler(filler_spec)
                return filler.vector
                
        elif isinstance(filler_spec, np.ndarray):
            # Direct vector specification
            return filler_spec
            
        elif isinstance(filler_spec, SymbolicStructure):
            # Nested structure
            return self.encode_structure(filler_spec)
            
        elif isinstance(filler_spec, dict):
            # Dictionary specification - create temporary structure
            temp_structure = SymbolicStructure(
                name=f"temp_{id(filler_spec)}",
                bindings=filler_spec
            )
            return self.encode_structure(temp_structure)
            
        elif isinstance(filler_spec, list):
            # List specification - encode as sequence
            return self._encode_sequence(filler_spec)
            
        else:
            # Default: convert to string and create filler
            filler_name = str(filler_spec)
            if filler_name not in self.filler_vocabulary:
                self.add_filler(filler_name)
            return self.filler_vocabulary[filler_name].vector
    
    def _encode_sequence(self, sequence: List[Any]) -> np.ndarray:
        """
        Encode a sequence using positional roles
        
        Args:
            sequence: List of elements to encode
            
        Returns:
            Vector encoding of sequence
        """
        if not sequence:
            return np.zeros(self.vector_dim)
        
        binding_pairs = []
        
        # Create positional roles
        for i, element in enumerate(sequence):
            # Use positional role
            position_role_name = f"POS_{i}"
            if position_role_name not in self.role_vocabulary:
                self.add_role(position_role_name, semantic_type="positional")
            
            role_vector = self.role_vocabulary[position_role_name].vector
            filler_vector = self._resolve_filler(element)
            
            binding_pair = BindingPair(role=role_vector, filler=filler_vector)
            binding_pairs.append(binding_pair)
        
        # Also bind length information
        length_role_name = "LENGTH"
        if length_role_name not in self.role_vocabulary:
            self.add_role(length_role_name, semantic_type="meta")
            
        length_filler_name = f"LEN_{len(sequence)}"
        if length_filler_name not in self.filler_vocabulary:
            self.add_filler(length_filler_name, semantic_type="numeric")
        
        length_binding = BindingPair(
            role=self.role_vocabulary[length_role_name].vector,
            filler=self.filler_vocabulary[length_filler_name].vector
        )
        binding_pairs.append(length_binding)
        
        return self.binder.bind_multiple(binding_pairs)
    
    def decode_structure(self, 
                        encoded_vector: np.ndarray,
                        known_roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Attempt to decode symbolic structure from vector
        
        Args:
            encoded_vector: Encoded structure vector
            known_roles: List of roles to try decoding (all if None)
            
        Returns:
            Dictionary of decoded role -> filler mappings
        """
        if known_roles is None:
            known_roles = list(self.role_vocabulary.keys())
        
        decoded_bindings = {}
        
        for role_name in known_roles:
            role = self.role_vocabulary[role_name]
            
            # Attempt to unbind using this role
            try:
                filler_vector = self.binder.unbind(encoded_vector, role.vector)
                
                # Find closest matching filler
                best_match = self._find_closest_filler(filler_vector)
                if best_match:
                    decoded_bindings[role_name] = best_match
                    
            except Exception:
                # Unbinding failed for this role
                continue
        
        return decoded_bindings
    
    def _find_closest_filler(self, 
                           target_vector: np.ndarray,
                           similarity_threshold: float = 0.5) -> Optional[str]:
        """Find closest matching filler in vocabulary"""
        best_similarity = -1
        best_match = None
        
        for filler_name, filler in self.filler_vocabulary.items():
            similarity = np.dot(target_vector, filler.vector)
            
            if similarity > best_similarity and similarity > similarity_threshold:
                best_similarity = similarity
                best_match = filler_name
        
        return best_match
    
    def encode_sentence(self, 
                       words: List[str],
                       structure_type: str = "linear") -> np.ndarray:
        """
        Encode a sentence using syntactic roles
        
        Args:
            words: List of words in sentence
            structure_type: Type of structure ("linear", "syntactic", "semantic")
            
        Returns:
            Vector encoding of sentence
        """
        if structure_type == "linear":
            # Simple positional encoding
            return self._encode_sequence(words)
            
        elif structure_type == "syntactic":
            # Use syntactic roles (simplified)
            if len(words) >= 3:
                structure = SymbolicStructure(
                    name=f"sentence_{id(words)}",
                    bindings={
                        "SUBJECT": words[0],
                        "VERB": words[1], 
                        "OBJECT": words[2] if len(words) > 2 else None
                    }
                )
                return self.encode_structure(structure)
            else:
                return self._encode_sequence(words)
                
        elif structure_type == "semantic":
            # Semantic role labeling (placeholder)
            return self._encode_sequence(words)
            
        else:
            return self._encode_sequence(words)
    
    def encode_tree(self, tree_dict: Dict[str, Any]) -> np.ndarray:
        """
        Encode a tree structure
        
        Args:
            tree_dict: Dictionary representing tree {"node": value, "children": [...]}
            
        Returns:
            Vector encoding of tree
        """
        if "node" not in tree_dict:
            return np.zeros(self.vector_dim)
        
        bindings = {"HEAD": tree_dict["node"]}
        
        # Handle children
        if "children" in tree_dict and tree_dict["children"]:
            children = tree_dict["children"]
            
            if len(children) == 1:
                bindings["CHILD"] = self.encode_tree(children[0])
            elif len(children) == 2:
                bindings["LEFT"] = self.encode_tree(children[0])
                bindings["RIGHT"] = self.encode_tree(children[1])
            else:
                # Multiple children - encode as sequence
                child_encodings = [self.encode_tree(child) for child in children]
                bindings["CHILDREN"] = self._encode_sequence(child_encodings)
        
        structure = SymbolicStructure(
            name=f"tree_{id(tree_dict)}",
            bindings=bindings,
            structure_type="tree"
        )
        
        return self.encode_structure(structure)
    
    def encode_predicate(self, 
                        predicate_name: str,
                        arguments: List[Any]) -> np.ndarray:
        """
        Encode a predicate with arguments
        
        Args:
            predicate_name: Name of predicate
            arguments: List of arguments
            
        Returns:
            Vector encoding of predicate
        """
        bindings = {"PRED": predicate_name}
        
        # Add arguments with positional roles
        for i, arg in enumerate(arguments):
            arg_role = f"ARG{i+1}"
            bindings[arg_role] = arg
        
        structure = SymbolicStructure(
            name=f"{predicate_name}({','.join(map(str, arguments))})",
            bindings=bindings,
            structure_type="predicate"
        )
        
        return self.encode_structure(structure)
    
    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Compute similarity between two structure vectors"""
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8)
    
    def get_structure_similarity(self, structure1_name: str, structure2_name: str) -> float:
        """Get similarity between two encoded structures"""
        if structure1_name in self.encoded_structures and structure2_name in self.encoded_structures:
            vec1 = self.encoded_structures[structure1_name]
            vec2 = self.encoded_structures[structure2_name]
            return self.similarity(vec1, vec2)
        return 0.0
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get statistics about the current vocabulary"""
        return {
            "n_roles": len(self.role_vocabulary),
            "n_fillers": len(self.filler_vocabulary),
            "n_encoded_structures": len(self.encoded_structures),
            "vector_dim": self.vector_dim,
            "role_types": {role.semantic_type for role in self.role_vocabulary.values()},
            "filler_types": {filler.semantic_type for filler in self.filler_vocabulary.values()}
        }