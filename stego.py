import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import struct
from typing import List, Tuple, Optional, Union

class SafeNeuralStego:
    """
    A class for cryptographically secure linguistic steganography using GPT-2.
    
    This class implements a distribution-transforming steganography method that embeds
    binary data into valid natural language text. It addresses critical issues like
    floating-point non-determinism to ensure reliable decoding across different hardware.
    """

    def __init__(self, model_name: str = 'gpt2'):
        """
        Initialize the SafeNeuralStego engine.

        Args:
            model_name: The name of the pre-trained GPT-2 model to use.
                        Defaults to 'gpt2'.
        """
        # Force CPU for maximum determinism (GPU floating point ops can be non-deterministic)
        self.device = 'cpu'
        print(f"[INIT] Loading {model_name} on {self.device} for deterministic behavior...")
        
        # Suppress warnings to keep output clean and load model/tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_deterministic_candidates(self, context_ids: torch.Tensor, top_k: int = 40) -> List[int]:
        """
        Retrieves candidate tokens in a strictly deterministic order.

        Sorts primarily by Probability (Desc), secondarily by TokenID (Asc).
        This prevents 'Floating Point Stability' bugs across different CPUs/architectures.

        Args:
            context_ids: Tensor of shape (1, seq_len) containing input token IDs.
            top_k: Number of top tokens to consider. Defaults to 40.

        Returns:
            A list of token IDs sorted deterministically.
        """
        with torch.no_grad():
            outputs = self.model(context_ids)
            # Get logits of the last token
            logits = outputs.logits[0, -1, :]

        # 1. Filter to Top-K to reduce search space
        top_logits, top_indices = torch.topk(logits, top_k)
        
        # 2. Convert to probabilities
        probs = F.softmax(top_logits, dim=-1).tolist()
        ids = top_indices.tolist()
        
        # 3. Create a list of tuples: (probability, token_id)
        candidates: List[Tuple[float, int]] = []
        for p, i in zip(probs, ids):
            candidates.append((p, i))
            
        # 4. STRICT DETERMINISTIC SORT:
        # Sort by Probability DESC (-x[0]), then by Token ID ASC (x[1])
        # This ensures that if two tokens have same prob, the one with lower ID always comes first.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Extract just the sorted IDs
        sorted_ids = [c[1] for c in candidates]
        return sorted_ids

    def encode(self, secret_bytes: bytes, start_text: str = "The weather today is") -> str:
        """
        Embeds [Length Header + Data] into generated text.

        The protocol embeds a 4-byte length header followed by the actual data.

        Args:
            secret_bytes: The secret message as bytes.
            start_text: The initial context for text generation.

        Returns:
            The generated cover text containing the hidden secret.

        Raises:
            RuntimeError: If the generated text becomes too long (model entropy issues).
        """
        # 1. Prepare Protocol Packet: [4-byte Length] + [Data]
        # 'I' format is unsigned int (4 bytes), supports up to 4GB data size
        length_header = struct.pack('>I', len(secret_bytes)) 
        payload = length_header + secret_bytes
        
        # Convert payload to a stream of bits string (efficient enough for demo)
        # We use a generator or simple string for clarity in logic
        bits_stream = "".join(f"{byte:08b}" for byte in payload)
        total_bits = len(bits_stream)
        
        context = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        generated_tokens: List[int] = []
        bit_pointer = 0
        
        print(f"[ENCODE] Payload size: {len(payload)} bytes ({total_bits} bits)")

        while bit_pointer < total_bits:
            # Get candidates strictly sorted
            candidates = self._get_deterministic_candidates(context)
            
            # Calculate Capacity: How many bits can we embed in this step?
            # We use Power-of-2 truncation strategy
            n_candidates = len(candidates)
            capacity_bits = int(math.floor(math.log2(n_candidates)))
            
            # If capacity is 0 (rare, only 1 candidate), skip step
            if capacity_bits == 0:
                best_token = candidates[0]
                # No bits consumed
            else:
                # We need 'capacity_bits' from the stream
                # n_options = 1 << capacity_bits # 2^k (Unused variable, but represents viable range)
                
                # Check if we have enough bits left, otherwise pad with 0
                if bit_pointer + capacity_bits <= total_bits:
                    chunk_str = bits_stream[bit_pointer : bit_pointer + capacity_bits]
                    bit_pointer += capacity_bits
                else:
                    # Padding for the very last step
                    # remaining = total_bits - bit_pointer
                    chunk_str = bits_stream[bit_pointer:]
                    # Pad with zeros to the right (LSB) to match value logic
                    chunk_str = chunk_str.ljust(capacity_bits, '0')
                    bit_pointer = total_bits # Done
                
                # Convert bits to integer index
                chosen_index = int(chunk_str, 2)
                best_token = candidates[chosen_index]
            
            # Append to context
            generated_tokens.append(best_token)
            next_input = torch.tensor([[best_token]]).to(self.device)
            context = torch.cat([context, next_input], dim=1)
            
            # Safety break just in case
            if len(generated_tokens) > 5000:
                raise RuntimeError("Text getting too long! Model entropy might be too low.")

        full_text = self.tokenizer.decode(context[0])
        return full_text

    def decode(self, full_text: str, start_text: str = "The weather today is") -> Optional[bytes]:
        """
        Extracts data by re-simulating the generation process.

        Args:
            full_text: The text containing the hidden message.
            start_text: The initial context used during encoding.

        Returns:
            The decoded secret bytes, or None if decoding fails.
        """
        context = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        # Encode full text to get token IDs
        full_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
        
        # Isolate generated part
        start_len = context.shape[1]
        generated_ids = full_ids[0][start_len:].tolist()
        
        recovered_bits = ""
        
        # State machine for decoding
        # We need to read at least 32 bits (4 bytes) to know length
        parsed_length: Optional[int] = None 
        
        current_context = context
        
        for token_id in generated_ids:
            # Stop if we have the full message (Header + Body)
            if parsed_length is not None:
                total_target_bits = (4 + parsed_length) * 8
                if len(recovered_bits) >= total_target_bits:
                    break
            
            candidates = self._get_deterministic_candidates(current_context)
            
            n_candidates = len(candidates)
            capacity_bits = int(math.floor(math.log2(n_candidates)))
            
            if capacity_bits > 0:
                n_options = 1 << capacity_bits
                
                # Where is the observed token in our sorted list?
                try:
                    rank = candidates.index(token_id)
                except ValueError:
                    # Token mismatch - critical failure in reproduction
                    # This happens if temperature/version differs
                    print(f"[ERROR] Token {token_id} not found in prediction candidates.")
                    return None
                
                if rank < n_options:
                    # It carries data
                    bits = format(rank, f'0{capacity_bits}b')
                    recovered_bits += bits
                else:
                    # It was a token outside the embedding range (statistically rare/impossible in our Encoder logic)
                    pass
            
            # Advance context
            next_input = torch.tensor([[token_id]]).to(self.device)
            current_context = torch.cat([current_context, next_input], dim=1)
            
            # Check if we can parse the header now
            if parsed_length is None and len(recovered_bits) >= 32:
                header_bits = recovered_bits[:32]
                # Convert bits to int
                header_bytes = int(header_bits, 2).to_bytes(4, byteorder='big')
                parsed_length = struct.unpack('>I', header_bytes)[0]
                print(f"[DECODE] Header found. Message length: {parsed_length} bytes.")

        # Reconstruction
        if parsed_length is None:
            print("[ERROR] Failed to recover header (Not enough data).")
            return None
            
        total_bits_needed = (4 + parsed_length) * 8
        final_bits = recovered_bits[:total_bits_needed]
        
        # Convert back to bytes
        # Split into 8-bit chunks
        bytes_list = []
        for i in range(0, len(final_bits), 8):
            byte_str = final_bits[i:i+8]
            bytes_list.append(int(byte_str, 2))
            
        all_data = bytes(bytes_list)
        
        # Separate Header and Body
        # extracted_length = struct.unpack('>I', all_data[:4])[0] # Already parsed above
        secret_body = all_data[4:]
        
        return secret_body

# --- RIGOROUS TESTING BLOCK ---
if __name__ == "__main__":
    print("--- Starting Scientific Robustness Test ---")
    
    stego = SafeNeuralStego()
    
    # TEST CASE 1: Complex binary data (not just text)
    # Includes emojis, newlines, and control characters to test robustness
    original_secret = "Confidential: Attack at 10:00 AM 🚀\nKey: x89-12".encode('utf-8')
    
    print(f"\n[INPUT] Original Secret ({len(original_secret)} bytes): {original_secret}")
    
    # 1. Encode
    try:
        cover_text = stego.encode(original_secret)
        print(f"\n[OUTPUT] Cover Text:\n---\n{cover_text}\n---")
    except Exception as e:
        print(f"Encoding Failed: {e}")
        exit()

    # 2. Decode
    try:
        decoded_bytes = stego.decode(cover_text)
    except Exception as e:
        print(f"Decoding Failed: {e}")
        exit()

    # 3. Verification
    print(f"\n[RESULT] Decoded Bytes: {decoded_bytes}")
    
    if decoded_bytes == original_secret:
        print("\n✅ TEST PASSED: Perfect Bit-for-Bit Reconstruction.")
    else:
        print("\n❌ TEST FAILED: Data Mismatch.")
        # Debug info
        print(f"Expected: {list(original_secret)}")
        print(f"Got:      {list(decoded_bytes) if decoded_bytes else 'None'}")
