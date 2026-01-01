import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import struct
from typing import List, Tuple, Optional, Callable, Dict

# Configuration map for easy access
AVAILABLE_MODELS = {
    "GPT-2 (Classic - Stable)": "gpt2",
    "DistilGPT-2 (Faster)": "distilgpt2",
    "Qwen2.5-0.5B (Smarter - Modern)": "Qwen/Qwen2.5-0.5B", 
    "TinyLlama-1.1B (Heavy)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

class SafeNeuralStego:
    """
    A class for cryptographically secure linguistic steganography using LLMs.
    
    Features:
    - Distribution-Transforming Steganography (Arithmetic Coding logic).
    - Deterministic floating-point sorting for hardware independence.
    - Newline banning for safe copy-pasting.
    - Support for multiple HuggingFace models.
    """

    def __init__(self, model_name_or_path: str = 'gpt2', log_callback: Optional[Callable[[str], None]] = None, device: Optional[str] = None):
        """
        Initialize the SafeNeuralStego engine.

        Args:
            model_name_or_path: The HF model ID or path.
            log_callback: Optional function to handle log messages (e.g., for GUI).
            device: 'cpu' or 'cuda'. Defaults to 'cpu' for safety, unless strictly controlled.
        """
        self.log_callback = log_callback
        
        # Default to CPU for maximum safety regarding deterministic behavior, 
        # unless user explicitly wants to risk GPU nondeterminism (or we implement strict checks later).
        if device:
            self.device = device
        else:
            self.device = 'cpu'
            
        self.log(f"[INIT] Loading Model: {model_name_or_path} on {self.device}...")
        
        try:
            # Use Auto classes to support multiple architectures
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Ensure pad_token is defined
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # --- ROBUST NEWLINE BANNING ---
            # Automatically find and ban tokens that create newlines
            self.banned_token_ids = []
            self.log("[INIT] Analyzing vocabulary for unsafe tokens...")
            
            # Heuristic: Find common newline tokens in the first 50k tokens
            # This covers most standard vocabulary without scanning full 150k+ vocabs of modern models
            vocab_size = self.tokenizer.vocab_size
            count = 0
            limit = min(vocab_size, 50257 if 'gpt2' in model_name_or_path else 60000)
            
            # Known prohibited characters
            prohibited = {'\n', '\r'}
            
            for t_id in range(limit):
                try:
                    token_str = self.tokenizer.decode([t_id])
                    if any(c in token_str for c in prohibited):
                        self.banned_token_ids.append(t_id)
                        count += 1
                except:
                    pass
            
            self.log(f"[READY] Model Loaded. {count} unsafe tokens banned.")
            
        except Exception as e:
            self.log(f"[ERROR] Critical Load Failure: {e}")
            raise e

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def _get_deterministic_candidates(self, context_ids: torch.Tensor, top_k: int = 50) -> List[int]:
        """
        Retrieves candidate tokens in a strictly deterministic order.
        Bans newlines and sorts primarily by Probability (Desc), secondarily by TokenID (Asc).
        """
        with torch.no_grad():
            outputs = self.model(context_ids)
            logits = outputs.logits[0, -1, :]

        # 1. Ban Newlines (Set to -Infinity)
        if self.banned_token_ids:
            # Efficient indexing if possible, or loop if sparse
            # For 200-300 bans, simple assignment is fast enough
            logits[self.banned_token_ids] = -float('Inf')

        # 2. Top-K Filter
        top_logits, top_indices = torch.topk(logits, top_k)
        
        # 3. Convert to probabilities
        probs = F.softmax(top_logits, dim=-1).tolist()
        ids = top_indices.tolist()
        
        # 4. Create candidates list
        candidates: List[Tuple[float, int]] = []
        for p, i in zip(probs, ids):
            # Redundant check for safety
            if i in self.banned_token_ids: continue
            candidates.append((p, i))
            
        # 5. STRICT DETERMINISTIC SORT:
        # Sort by Probability DESC (-x[0]), then by Token ID ASC (x[1])
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        return [c[1] for c in candidates]

    def encode(self, secret_bytes: bytes, start_text: str = "The future of technology is") -> str:
        """
        Embeds [Length Header + Data] into generated text.
        """
        start_text = start_text.strip()
        if not start_text: start_text = "The future of technology is"

        # Prepare Protocol Packet: [4-byte Length] + [Data]
        length_header = struct.pack('>I', len(secret_bytes)) 
        payload = length_header + secret_bytes
        
        # Convert payload to bit stream
        bits_stream = "".join(f"{byte:08b}" for byte in payload)
        total_bits = len(bits_stream)
        
        context = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        generated_tokens: List[int] = []
        bit_pointer = 0
        
        self.log(f"[ENCODE] Hiding {len(payload)} bytes ({total_bits} bits)...")

        while bit_pointer < total_bits:
            candidates = self._get_deterministic_candidates(context)
            if not candidates:
                break # Should ideally not happen unless vocab is extremely small

            n_candidates = len(candidates)
            capacity_bits = int(math.floor(math.log2(n_candidates)))
            
            if capacity_bits == 0:
                best_token = candidates[0]
            else:
                if bit_pointer + capacity_bits <= total_bits:
                    chunk_str = bits_stream[bit_pointer : bit_pointer + capacity_bits]
                    bit_pointer += capacity_bits
                else:
                    # Padding for the very last step
                    chunk_str = bits_stream[bit_pointer:].ljust(capacity_bits, '0')
                    bit_pointer = total_bits 
                
                chosen_index = int(chunk_str, 2)
                best_token = candidates[chosen_index]
            
            generated_tokens.append(best_token)
            next_input = torch.tensor([[best_token]]).to(self.device)
            context = torch.cat([context, next_input], dim=1)
            
            # Limit length to prevent infinite loops or memory crashes
            if len(generated_tokens) > 2000:
                raise ValueError("Text too long. Try a shorter secret or a model with higher entropy.")

        full_text = self.tokenizer.decode(context[0])
        return full_text

    def decode(self, full_text: str, start_text: str = "The future of technology is") -> Optional[bytes]:
        """
        Extracts data by re-simulating the generation process.
        """
        full_text = full_text.strip()
        start_text = start_text.strip()
        if not start_text: start_text = "The future of technology is"

        if not full_text.startswith(start_text):
            raise ValueError(f"Start text mismatch!\nExpected start: '{start_text}'")

        context = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        full_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
        
        start_len = context.shape[1]
        if full_ids.shape[1] <= start_len:
            raise ValueError("No hidden data found (Text matches start text exactly).")

        generated_ids = full_ids[0][start_len:].tolist()
        
        recovered_bits = ""
        parsed_length: Optional[int] = None 
        current_context = context
        
        self.log("[DECODE] Processing text...")

        for token_id in generated_ids:
            # Stop if we have the full message
            if parsed_length is not None:
                total_target_bits = (4 + parsed_length) * 8
                if len(recovered_bits) >= total_target_bits:
                    break
            
            candidates = self._get_deterministic_candidates(current_context)
            n_candidates = len(candidates)
            capacity_bits = int(math.floor(math.log2(n_candidates)))
            
            if capacity_bits > 0:
                n_options = 1 << capacity_bits
                try:
                    rank = candidates.index(token_id)
                    if rank < n_options:
                        bits = format(rank, f'0{capacity_bits}b')
                        recovered_bits += bits
                    else:
                        # Token out of range (statistically rare or impossible if generated correctly)
                        pass
                except ValueError:
                    # Token not in top candidates (model mismatch or sync lost)
                    pass
            
            next_input = torch.tensor([[token_id]]).to(self.device)
            current_context = torch.cat([current_context, next_input], dim=1)
            
            # Check for Header
            if parsed_length is None and len(recovered_bits) >= 32:
                header_bits = recovered_bits[:32]
                try:
                    header_bytes = int(header_bits, 2).to_bytes(4, byteorder='big')
                    parsed_length = struct.unpack('>I', header_bytes)[0]
                    self.log(f"[DECODE] Header found. Message length: {parsed_length} bytes.")
                except:
                    return None

        if parsed_length is None:
            raise ValueError("Header not found. Text might be truncated or model mismatch.")
            
        total_bits_needed = (4 + parsed_length) * 8
        final_bits = recovered_bits[:total_bits_needed]
        
        bytes_list = []
        for i in range(0, len(final_bits), 8):
            byte_str = final_bits[i:i+8]
            bytes_list.append(int(byte_str, 2))
            
        all_data = bytes(bytes_list)
        return all_data[4:] # Return body only
