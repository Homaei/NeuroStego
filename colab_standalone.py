# ==========================================
# NEUROSTEGO - GOOGLE COLAB STANDALONE EDITION
# ==========================================

# 1. Install dependencies automatically if running in cell
try:
    import google.colab
    IN_COLAB = True
    print("Installing dependencies...")
    get_ipython().system('pip install transformers torch accelerate > /dev/null 2>&1')
except ImportError:
    IN_COLAB = False

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import struct
import sys
import gc
from typing import List, Tuple, Optional

# --- THE LOGIC CORE ---
class SafeNeuralStegoLite:
    def __init__(self, model_name_or_path='gpt2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⏳ Loading Model: {model_name_or_path} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Heuristic Newline Banning for simple Colab usage
            self.banned_token_ids = []
            # Common newline tokens for GPT/Llama to ban
            prohibited = {'\n', '\r'}
            try:
                # Fast incomplete scan
                for t_id in range(min(50000, self.tokenizer.vocab_size)):
                    if any(c in self.tokenizer.decode([t_id]) for c in prohibited):
                        self.banned_token_ids.append(t_id)
            except:
                pass
            
            print(f"✅ Model Loaded. {len(self.banned_token_ids)} tokens banned for clean output.")
            
        except Exception as e:
            print(f"❌ Critical Load Failure: {e}")
            raise e

    def _get_candidates(self, context_ids, top_k=50):
        with torch.no_grad():
            outputs = self.model(context_ids)
            logits = outputs.logits[0, -1, :]

        if self.banned_token_ids:
            logits[self.banned_token_ids] = -float('Inf')

        top_logits, top_indices = torch.topk(logits, top_k)
        probs = F.softmax(top_logits, dim=-1).tolist()
        ids = top_indices.tolist()
        
        candidates = []
        for p, i in zip(probs, ids):
            if i in self.banned_token_ids: continue
            candidates.append((p, i))
            
        # Deterministic Sort
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return [c[1] for c in candidates]

    def encode(self, secret_bytes, start_text):
        start_text = start_text.strip()
        if not start_text: start_text = "The future of technology is"

        length_header = struct.pack('>I', len(secret_bytes)) 
        payload = length_header + secret_bytes
        bits_stream = "".join(f"{byte:08b}" for byte in payload)
        total_bits = len(bits_stream)
        
        context = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        generated_tokens = []
        bit_pointer = 0
        
        print(f"🔄 Encoding {len(payload)} bytes...")

        while bit_pointer < total_bits:
            candidates = self._get_candidates(context)
            if not candidates: break

            n_candidates = len(candidates)
            capacity_bits = int(math.floor(math.log2(n_candidates)))
            
            if capacity_bits == 0:
                best_token = candidates[0]
            else:
                if bit_pointer + capacity_bits <= total_bits:
                    chunk_str = bits_stream[bit_pointer : bit_pointer + capacity_bits]
                    bit_pointer += capacity_bits
                else:
                    chunk_str = bits_stream[bit_pointer:].ljust(capacity_bits, '0')
                    bit_pointer = total_bits 
                
                chosen_index = int(chunk_str, 2)
                best_token = candidates[chosen_index]
            
            generated_tokens.append(best_token)
            next_input = torch.tensor([[best_token]]).to(self.device)
            context = torch.cat([context, next_input], dim=1)
            
            if len(generated_tokens) > 2000:
                raise ValueError("Text too long.")

        return self.tokenizer.decode(context[0])

    def decode(self, full_text, start_text):
        full_text = full_text.strip()
        start_text = start_text.strip()
        if not start_text: start_text = "The future of technology is"

        context = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        full_ids = self.tokenizer.encode(full_text, return_tensors='pt').to(self.device)
        
        start_len = context.shape[1]
        try:
            generated_ids = full_ids[0][start_len:].tolist()
        except:
             raise ValueError("Text mismatch or too short.")
        
        recovered_bits = ""
        parsed_length = None 
        current_context = context
        
        print("🔄 Decoding...")

        for token_id in generated_ids:
            if parsed_length is not None:
                total_target_bits = (4 + parsed_length) * 8
                if len(recovered_bits) >= total_target_bits:
                    break
            
            candidates = self._get_candidates(current_context)
            n_candidates = len(candidates)
            capacity_bits = int(math.floor(math.log2(n_candidates)))
            
            if capacity_bits > 0:
                n_options = 1 << capacity_bits
                try:
                    rank = candidates.index(token_id)
                    if rank < n_options:
                        bits = format(rank, f'0{capacity_bits}b')
                        recovered_bits += bits
                except ValueError:
                    pass
            
            next_input = torch.tensor([[token_id]]).to(self.device)
            current_context = torch.cat([current_context, next_input], dim=1)
            
            if parsed_length is None and len(recovered_bits) >= 32:
                header_bits = recovered_bits[:32]
                try:
                    header_bytes = int(header_bits, 2).to_bytes(4, byteorder='big')
                    parsed_length = struct.unpack('>I', header_bytes)[0]
                except:
                    return None

        if parsed_length is None:
            raise ValueError("Header not found.")
            
        total_bits_needed = (4 + parsed_length) * 8
        final_bits = recovered_bits[:total_bits_needed]
        
        bytes_list = []
        for i in range(0, len(final_bits), 8):
            bytes_list.append(int(final_bits[i:i+8], 2))
            
        return bytes(bytes_list)[4:]

# --- INTERACTIVE LOOP ---
if __name__ == "__main__":
    print("--- NEUROSTEGO (COLAB MODE) ---")
    # Default to distilgpt2 for speed in free colab tiers
    stego = SafeNeuralStegoLite("distilgpt2") 
    
    while True:
        print("\n1. Encode")
        print("2. Decode")
        print("3. Exit")
        choice = input("Select: ")
        
        if choice == '1':
            secret = input("Secret Message: ")
            start = input("Start Text (Optional): ")
            try:
                res = stego.encode(secret.encode('utf-8'), start)
                print("\n--- COPY BELOW ---")
                print(res)
                print("------------------")
            except Exception as e: print(e)
            
        elif choice == '2':
            text = input("Paste Full Text: ")
            start = input("Original Start Text: ")
            try:
                res = stego.decode(text, start)
                print(f"\n🔓 SECRET: {res.decode('utf-8', errors='replace')}")
            except Exception as e: print(f"Error: {e}")
            
        elif choice == '3':
            break
