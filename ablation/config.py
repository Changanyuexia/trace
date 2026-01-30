"""
Configuration for ablation study modules.

Each module can be enabled/disabled independently to create different system variants.
"""
from dataclasses import dataclass
from typing import Dict, Any, Mapping, Optional

@dataclass
class AblationConfig:
    """Configuration for ablation study variants."""
    
    # Module flags
    enable_tdd_gate: bool = False          # G1: TDD Gate (RED/GREEN verification)
    enable_index_retrieval: bool = False   # G2/G5: tools_retrieval index retrieval
    enable_patch_compile_gate: bool = False # G3: Patch/Compile gate
    
    # G0 Baseline settings (always enabled)
    use_grep_read_file: bool = True        # Use grep/read_file for localization
    use_unified_diff: bool = True          # LLM outputs unified diff directly
    use_full_test_validation: bool = True  # Direct full test/eval validation
    
    # G1: TDD Gate settings
    verify_red_test: bool = False          # Verify RED test before localization
    verify_green_test: bool = False         # Verify GREEN test after patching
    
    # G2: Index Retrieval settings
    use_symbol_lookup: bool = False         # Use symbol_lookup tool
    use_find_references: bool = False      # Use find_references tool
    use_read_span: bool = False            # Use read_span tool
    max_symbol_blocks_per_round: int = 10  # Working set limit (N symbol blocks)
    
    # G3: Patch/Compile Gate settings
    use_git_apply_check: bool = False      # git apply --check before applying
    use_canonical_diff: bool = False        # Generate canonical diff (git diff)
    use_compile_gate: bool = False         # Compile gate (defects4j compile)
    
    # API call limits (to prevent infinite loops and high costs)
    max_localization_api_calls: int = 36  # Maximum API calls in localization phase
    max_patch_phase_api_calls: int = 50   # Maximum API calls in patch phase
    max_tool_calls_per_patch: int = 4     # Maximum tool calls per patch attempt (increased from 3 to allow more learning)
    max_consecutive_direct_patches: int = 5  # Maximum consecutive direct patch returns before stopping (infinite loop detection)
    max_git_apply_failures: int = 5        # Maximum consecutive git apply failures before stopping
    max_compile_failures: int = 5          # Maximum compilation failures before stopping (G3/G5)
    
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AblationConfig":
        """Create config from a dict (e.g. loaded from JSON). Unknown keys are ignored."""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(d).items() if k in allowed}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_tdd_gate": self.enable_tdd_gate,
            "enable_index_retrieval": self.enable_index_retrieval,
            "enable_patch_compile_gate": self.enable_patch_compile_gate,
            "use_grep_read_file": self.use_grep_read_file,
            "use_unified_diff": self.use_unified_diff,
            "use_full_test_validation": self.use_full_test_validation,
            "verify_red_test": self.verify_red_test,
            "verify_green_test": self.verify_green_test,
            "use_symbol_lookup": self.use_symbol_lookup,
            "use_find_references": self.use_find_references,
            "use_read_span": self.use_read_span,
            "max_symbol_blocks_per_round": self.max_symbol_blocks_per_round,
            "use_git_apply_check": self.use_git_apply_check,
            "use_canonical_diff": self.use_canonical_diff,
            "use_compile_gate": self.use_compile_gate,
            "max_localization_api_calls": self.max_localization_api_calls,
            "max_patch_phase_api_calls": self.max_patch_phase_api_calls,
            "max_tool_calls_per_patch": self.max_tool_calls_per_patch,
            "max_consecutive_direct_patches": self.max_consecutive_direct_patches,
            "max_git_apply_failures": self.max_git_apply_failures,
            "max_compile_failures": self.max_compile_failures,
        }
    
    @classmethod
    def from_variant(cls, variant: str) -> "AblationConfig":
        """
        Create config from variant name.
        
        Variants:
        - G0: Baseline
        - G1: G0 + TDD Gate
        - G2: G0 + Index Retrieval
        - G3: G0 + Patch/Compile Gate
        - G5: Full system (all enabled)
        """
        variant = variant.upper()
        
        if variant == "G0":
            # Baseline: only grep/read_file, unified diff, full test
            return cls(
                enable_tdd_gate=False,
                enable_index_retrieval=False,
                enable_patch_compile_gate=False,
            )
        elif variant == "G1":
            # G0 + TDD Gate
            return cls(
                enable_tdd_gate=True,
                enable_index_retrieval=False,
                enable_patch_compile_gate=False,
                verify_red_test=True,
                verify_green_test=True,
            )
        elif variant == "G2":
            # G0 + Index Retrieval
            return cls(
                enable_tdd_gate=False,
                enable_index_retrieval=True,
                enable_patch_compile_gate=False,
                use_symbol_lookup=True,
                use_find_references=True,
                use_read_span=True,
            )
        elif variant == "G3":
            # G0 + Patch/Compile Gate
            return cls(
                enable_tdd_gate=False,
                enable_index_retrieval=False,
                enable_patch_compile_gate=True,
                use_git_apply_check=True,
                use_canonical_diff=True,
                use_compile_gate=True,
            )
        elif variant == "G5":
            # Full system: all features enabled
            return cls(
                enable_tdd_gate=True,
                enable_index_retrieval=True,
                enable_patch_compile_gate=True,
                verify_red_test=True,
                verify_green_test=True,
                use_symbol_lookup=True,
                use_find_references=True,
                use_read_span=True,
                use_git_apply_check=True,
                use_canonical_diff=True,
                use_compile_gate=True,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}. Supported: G0, G1, G2, G3, G5")







