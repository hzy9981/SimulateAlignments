import os
import sys
import numpy as np
import random
from itertools import permutations
from fairseq.models.transformer import TransformerModel
from Bio import SeqIO

def encode_source(unaligned_seqs, perm):
    """Encodes sequences in a specific permutation."""
    ordered = [unaligned_seqs[i] for i in perm]
    source = " | ".join([" ".join(list(s)) for s in ordered])
    return source

def decode_target(target_str, num_seqs, perm):
    """Decodes the interleaved target string back into MSA rows, reordering to original."""
    tokens = target_str.split()
    if len(tokens) % num_seqs != 0:
        return None
    
    seq_len = len(tokens) // num_seqs
    msa_rows = [""] * num_seqs
    
    # Reconstruct the permuted MSA
    for i in range(seq_len):
        col = tokens[i*num_seqs : (i+1)*num_seqs]
        for j in range(num_seqs):
            msa_rows[j] += col[j]
            
    # Reorder back to original sequence indices
    original_msa = [""] * num_seqs
    for idx, original_pos in enumerate(perm):
        original_msa[original_pos] = msa_rows[idx]
        
    return original_msa

def get_consensus_msa(results, num_seqs):
    """Simple majority vote consensus for MSA columns."""
    # results is a list of MSA rows (list of strings)
    # We assume all valid results have the same column count if generated correctly, 
    # but in practice, they might vary. The paper uses a 'certainty' score.
    
    # For simplicity in this reproduction script, we take the most frequent column at each position
    # This is a simplified version of the paper's consensus logic.
    
    # First, we need to handle varying lengths or invalid alignments.
    # A better approach for reproduction is to calculate column frequency.
    
    # Map each result to a list of columns
    all_msas_cols = []
    for msa in results:
        cols = []
        for i in range(len(msa[0])):
            cols.append(tuple([msa[j][i] for j in range(num_seqs)]))
        all_msas_cols.append(cols)
    
    # The paper's logic is more complex (tracking character positions), 
    # but here we implement a basic version.
    return results[0] # Placeholder: return the first valid one

def run_inference(model_path, checkpoint, data_bin, fasta_in, num_perms=10):
    # Load model
    model = TransformerModel.from_pretrained(
        model_path,
        checkpoint_file=checkpoint,
        data_name_or_path=data_bin,
        max_len_a=6
    )
    
    records = list(SeqIO.parse(fasta_in, "fasta"))
    unaligned = [str(r.seq).upper().replace("-", "") for r in records]
    num_seqs = len(unaligned)
    
    # Generate permutations
    all_perms = list(permutations(range(num_seqs)))
    if len(all_perms) > num_perms:
        sampled_perms = random.sample(all_perms, num_perms)
    else:
        sampled_perms = all_perms
        
    results = []
    for perm in sampled_perms:
        src = encode_source(unaligned, perm)
        tgt_out = model.translate(src, beam=15)
        decoded = decode_target(tgt_out, num_seqs, perm)
        if decoded:
            # Check validity: unaligned characters must match
            valid = True
            for i in range(num_seqs):
                if decoded[i].replace("-", "") != unaligned[i]:
                    valid = False
                    break
            if valid:
                results.append(decoded)
    
    if not results:
        print("Error: No valid alignments generated.")
        return None
        
    # Return consensus (simplified)
    return results[0]

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python reproduce_inference.py <model_dir> <checkpoint> <data_bin> <input_fasta>")
        sys.exit(1)
        
    m_dir, m_chk, d_bin, f_in = sys.argv[1:5]
    msa = run_inference(m_dir, m_chk, d_bin, f_in)
    if msa:
        for i, row in enumerate(msa):
            print(f">Seq_{i}\n{row}")
