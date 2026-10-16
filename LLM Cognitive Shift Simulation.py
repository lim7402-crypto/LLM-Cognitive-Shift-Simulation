"""
LLM Reasoning Shift Simulation: Traditional Linear vs. Sparse Bifurcation Processing

This module simulates the structural transformation of Large Language Model (LLM) reasoning:
1. TraditionalLLMProcessor: Uniform, linear token processing with high compute cost and low SNR.
2. SparseBifurcationLLMProcessor: L1-norm soft thresholding for sparsity filtering combined
   with critical phase-transition (bifurcation point) detection for high density and low compute cost.
"""

import math
from typing import List, Dict, Any, Tuple


class TraditionalLLMProcessor:
    """
    Simulates standard LLM processing:
    Treats all input tokens uniformly without sparsity filtering, resulting in linear context
    accumulation, high computational overhead, and low Signal-to-Noise Ratio (SNR).
    """
    def __init__(self, raw_tokens: List[str], token_signals: Dict[str, float]):
        self.raw_tokens = raw_tokens
        self.token_signals = token_signals

    def process(self) -> Dict[str, Any]:
        num_tokens = len(self.raw_tokens)
        
        # Linear processing weights every token uniformly (no filtering)
        weights = {token: 1.0 for token in self.raw_tokens}
        
        # Calculate Signal-to-Noise Ratio (SNR)
        # Signal = meaningful variance, Noise = background fluff
        signal = sum(self.token_signals.get(t, 0.0) for t in self.raw_tokens)
        noise = sum(1.0 - self.token_signals.get(t, 0.0) for t in self.raw_tokens)
        snr_db = 10 * math.log10((signal / max(noise, 1e-5)))

        # Computational cost scales quadratically/linearly with unfiltered sequence length
        compute_cost_flops = num_tokens ** 2 * 1000  # Arbitrary FLOPs scaling unit
        
        # Information density = Signal per processed token
        info_density = signal / num_tokens

        return {
            "mode": "Traditional Linear LLM Processing",
            "processed_token_count": num_tokens,
            "sparsity_ratio": 0.0,  # 0% pruned
            "snr_db": round(snr_db, 2),
            "info_density": round(info_density, 3),
            "compute_cost_flops": compute_cost_flops,
            "bifurcation_leap": False,
            "status": "Linear context accumulation complete."
        }


class SparseBifurcationLLMProcessor:
    """
    Simulates LLM reasoning under L1-norm sparsity filtering and bifurcation dynamic framework:
    Applies soft-thresholding to eliminate noise tokens and detects critical phase transitions
    to execute high-density causal reasoning leaps with dramatic computational savings.
    """
    def __init__(self, raw_tokens: List[str], threshold: float = 0.5):
        self.raw_tokens = raw_tokens
        self.threshold = threshold

    def _l1_soft_thresholding(self, token_signals: Dict[str, float]) -> Dict[str, float]:
        """Applies L1-norm soft-thresholding operator: S_lambda(x) = sign(x) * max(|x| - lambda, 0)"""
        sparse_signals = {}
        for token, signal in token_signals.items():
            sparse_val = max(0.0, signal - self.threshold)
            if sparse_val > 0.0:
                sparse_signals[token] = sparse_val
        return sparse_signals

    def _detect_bifurcation(self, sparse_signals: Dict[str, float]) -> Tuple[bool, float]:
        """Detects whether signal density crosses the critical boundary for a phase transition leap."""
        if not sparse_signals:
            return False, 0.0
        
        total_sparse_energy = sum(sparse_signals.values())
        sparse_density = total_sparse_energy / len(sparse_signals)
        critical_bifurcation_threshold = 0.25

        has_bifurcated = sparse_density >= critical_bifurcation_threshold
        return has_bifurcated, sparse_density

    def process(self, token_signals: Dict[str, float]) -> Dict[str, Any]:
        num_raw_tokens = len(self.raw_tokens)

        # 1. Apply L1-norm sparsity filtering
        sparse_tokens = self._l1_soft_thresholding(token_signals)
        num_retained_tokens = len(sparse_tokens)
        num_pruned_tokens = num_raw_tokens - num_retained_tokens
        sparsity_ratio = num_pruned_tokens / max(num_raw_tokens, 1)

        # 2. Calculate filtered SNR
        signal = sum(sparse_tokens.values())
        noise = sum(self.threshold for _ in range(num_retained_tokens))
        snr_db = 10 * math.log10((signal / max(noise, 1e-5)))

        # 3. Detect Phase Transition / Bifurcation Point
        has_bifurcated, sparse_density = self._detect_bifurcation(sparse_tokens)

        # 4. Compute cost scales on pruned sequence length
        compute_cost_flops = (num_retained_tokens ** 2) * 1000

        # Information density multiplied by bifurcation leap multiplier
        leap_multiplier = 2.5 if has_bifurcated else 1.0
        info_density = (signal / max(num_retained_tokens, 1)) * leap_multiplier

        return {
            "mode": "Sparse Bifurcation LLM Processing",
            "processed_token_count": num_retained_tokens,
            "pruned_token_count": num_pruned_tokens,
            "sparsity_ratio": round(sparsity_ratio * 100, 1),
            "snr_db": round(snr_db, 2),
            "info_density": round(info_density, 3),
            "compute_cost_flops": compute_cost_flops,
            "bifurcation_leap": has_bifurcated,
            "retained_anchors": list(sparse_tokens.keys()),
            "status": "Phase-transition causal leap executed." if has_bifurcated else "Sub-critical linear response."
        }


def print_comparison_benchmark(traditional_res: Dict[str, Any], sparse_res: Dict[str, Any]):
    """Prints a clean ASCII side-by-side comparative metric dashboard."""
    print("=" * 80)
    print("      LLM REASONING SHIFT BENCHMARK: TRADITIONAL VS. SPARSE BIFURCATION      ")
    print("=" * 80)
    print(f"{'Metric / Parameter':<30} | {'Traditional LLM':<22} | {'Sparse Bifurcation':<22}")
    print("-" * 80)
    
    print(f"{'Processing Mode':<30} | {'Linear Uniform':<22} | {'L1-Norm + Phase Shift':<22}")
    print(f"{'Processed Tokens':<30} | {str(traditional_res['processed_token_count']):<22} | {str(sparse_res['processed_token_count']):<22}")
    print(f"{'Sparsity / Pruning Ratio':<30} | {f\"{traditional_res['sparsity_ratio']:.1f}%\":<22} | {f\"{sparse_res['sparsity_ratio']:.1f}%\":<22}")
    print(f"{'Signal-to-Noise Ratio (SNR)':<30} | {f\"{traditional_res['snr_db']} dB\":<22} | {f\"{sparse_res['snr_db']} dB\":<22}")
    print(f"{'Information Density Score':<30} | {str(traditional_res['info_density']):<22} | {str(sparse_res['info_density']):<22}")
    print(f"{'Compute Cost (FLOPs)':<30} | {f\"{traditional_res['compute_cost_flops']:,}\":<22} | {f\"{sparse_res['compute_cost_flops']:,}\":<22}")
    print(f"{'Phase Transition Leap':<30} | {str(traditional_res['bifurcation_leap']):<22} | {str(sparse_res['bifurcation_leap']):<22}")
    
    # Calculate performance improvements
    compute_reduction = ((traditional_res['compute_cost_flops'] - sparse_res['compute_cost_flops']) 
                         / traditional_res['compute_cost_flops']) * 100
    density_gain = ((sparse_res['info_density'] - traditional_res['info_density']) 
                    / traditional_res['info_density']) * 100

    print("-" * 80)
    print(" PERFORMANCE SUMMARY:")
    print(f"  • Computational Overhead Saved : {compute_reduction:.1f}%")
    print(f"  • Information Density Gain    : +{density_gain:.1f}%")
    print(f"  • Retained Sparse Anchors     : {', '.join(sparse_res['retained_anchors'])}")
    print("=" * 80)


if __name__ == "__main__":
    # Sample input tokens containing noisy background text and key sparse signals
    raw_prompt_tokens = [
        "preamble_fluff_1",
        "continuous_topological_space",
        "filler_phrase_2",
        "sparsity_l1_norm",
        "conversational_padding",
        "bifurcation_dynamics",
        "redundant_clause_3",
        "phase_transition_point"
    ]

    # Meaningful signal relevance scores (range: 0.0 to 1.0)
    token_relevance = {
        "preamble_fluff_1": 0.08,
        "continuous_topological_space": 0.88,
        "filler_phrase_2": 0.12,
        "sparsity_l1_norm": 0.96,
        "conversational_padding": 0.05,
        "bifurcation_dynamics": 0.92,
        "redundant_clause_3": 0.15,
        "phase_transition_point": 0.94
    }

    # Run Traditional LLM Simulation
    trad_processor = TraditionalLLMProcessor(raw_prompt_tokens, token_relevance)
    traditional_results = trad_processor.process()

    # Run Sparse Bifurcation LLM Simulation
    sparse_processor = SparseBifurcationLLMProcessor(raw_prompt_tokens, threshold=0.50)
    sparse_results = sparse_processor.process(token_relevance)

    # Output Benchmark Results
    print_comparison_benchmark(traditional_results, sparse_results)