import customtkinter as ctk
import threading
import torch
import gc
import pyperclip
import webbrowser
from stego import SafeNeuralStego, AVAILABLE_MODELS

# UI Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class NeuroStegoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NeuroStego v2.0 - Multi-Model Edition")
        self.geometry("800x700")
        
        self.stego = None
        self.current_model_id = None
        
        # Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Header Area
        self.frame_header = ctk.CTkFrame(self)
        self.frame_header.grid(row=0, column=0, padx=20, pady=15, sticky="ew")
        
        self.lbl_title = ctk.CTkLabel(self.frame_header, text="NeuroStego", font=("Roboto Medium", 24))
        self.lbl_title.pack(side="left", padx=20, pady=10)
        
        # Model Selector
        self.lbl_model = ctk.CTkLabel(self.frame_header, text="AI Model:")
        self.lbl_model.pack(side="left", padx=(20, 5))
        
        self.combo_model = ctk.CTkComboBox(
            self.frame_header, 
            values=list(AVAILABLE_MODELS.keys()),
            width=250,
            command=self.on_model_change
        )
        self.combo_model.pack(side="left", padx=5)
        self.combo_model.set("GPT-2 (Classic - Stable)") # Default

        # 2. Main Tab View
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=2, column=0, padx=20, pady=0, sticky="nsew")
        
        self.tab_enc = self.tabview.add("ENCODE")
        self.tab_dec = self.tabview.add("DECODE")
        self.tab_about = self.tabview.add("ABOUT")
        
        self.setup_encode_tab()
        self.setup_decode_tab()
        self.setup_about_tab()

        # 3. Status Bar
        self.frame_status = ctk.CTkFrame(self, height=30)
        self.frame_status.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.status_bar = ctk.CTkLabel(self.frame_status, text="System initialized.", text_color="silver", anchor="w")
        self.status_bar.pack(side="left", padx=10, pady=5)

        # Initial Load
        self.on_model_change("GPT-2 (Classic - Stable)")

    def setup_encode_tab(self):
        t = self.tab_enc
        t.grid_columnconfigure(0, weight=1)
        
        # Row 0: Secret Input
        ctk.CTkLabel(t, text="Secret Message:", anchor="w", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=10, pady=(15,5), sticky="w")
        self.entry_secret = ctk.CTkEntry(t, placeholder_text="Type your secret message here to hide it...", height=40)
        self.entry_secret.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Row 2: Context Input
        ctk.CTkLabel(t, text="Start Text (Context):", anchor="w", font=("Arial", 14, "bold")).grid(row=2, column=0, padx=10, pady=(15,5), sticky="w")
        ctk.CTkLabel(t, text="The AI will continue this sentence.", text_color="gray", anchor="w").grid(row=2, column=0, padx=160, pady=(15,5), sticky="w")
        
        self.entry_start_enc = ctk.CTkEntry(t, placeholder_text="e.g. The weather today is likely to be")
        self.entry_start_enc.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        self.entry_start_enc.insert(0, "The future of technology is")

        # Row 4: Action Button
        self.btn_encode = ctk.CTkButton(t, text="🔒 Generate Cloaked Text", command=self.run_encode, state="disabled", height=40, font=("Arial", 14, "bold"))
        self.btn_encode.grid(row=4, column=0, padx=10, pady=25, sticky="ew")

        # Row 5: Output
        ctk.CTkLabel(t, text="Generated Cover Text (Safe to send):", anchor="w").grid(row=5, column=0, padx=10, pady=(5,0), sticky="w")
        self.txt_out_enc = ctk.CTkTextbox(t, height=150)
        self.txt_out_enc.grid(row=6, column=0, padx=10, pady=5, sticky="nsew")
        
        # Row 7: Copy Button
        self.btn_copy = ctk.CTkButton(t, text="Copy to Clipboard", fg_color="gray", command=self.copy_enc)
        self.btn_copy.grid(row=7, column=0, padx=10, pady=10, sticky="e")

    def setup_decode_tab(self):
        t = self.tab_dec
        t.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(t, text="Paste Cloaked Text:", anchor="w", font=("Arial", 14, "bold")).grid(row=0, column=0, padx=10, pady=(15,5), sticky="w")
        self.txt_in_dec = ctk.CTkTextbox(t, height=150)
        self.txt_in_dec.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(t, text="Start Text (Must Match Exactly):", anchor="w", font=("Arial", 14, "bold")).grid(row=2, column=0, padx=10, pady=(15,5), sticky="w")
        self.entry_start_dec = ctk.CTkEntry(t, placeholder_text="Enter the start text used for encoding...")
        self.entry_start_dec.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.btn_decode = ctk.CTkButton(t, text="🔓 Reveal Secret", fg_color="#2CC985", hover_color="#229965", command=self.run_decode, state="disabled", height=40, font=("Arial", 14, "bold"))
        self.btn_decode.grid(row=4, column=0, padx=10, pady=25, sticky="ew")
        
        ctk.CTkLabel(t, text="Decoded Secret:", anchor="w", font=("Arial", 14, "bold")).grid(row=5, column=0, padx=10, sticky="w")
        
        self.txt_out_dec = ctk.CTkTextbox(t, height=80, text_color="#2CC985", font=("Consolas", 14))
        self.txt_out_dec.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        
    def setup_about_tab(self):
        t = self.tab_about
        content = """
        NeuroStego
        Cryptographically Secure Linguistic Steganography
        
        This tool hides data within the probability distribution of a Language Model.
        
        Features:
        - No metadata leakage
        - No statistical anomalies
        - Deterministic decoding across CPUs
        
        Author: Homaei / DeepMind Assistant
        License: MIT
        """
        lbl = ctk.CTkLabel(t, text=content, justify="left", font=("Arial", 14))
        lbl.pack(padx=20, pady=20)
        
        btn = ctk.CTkButton(t, text="View on GitHub", command=lambda: webbrowser.open("https://github.com/Homaei/NeuroStego"))
        btn.pack(pady=10)

    def update_status(self, msg):
        # Schedule update on main thread if called from worker
        self.status_bar.configure(text=msg)
        print(msg) 

    def on_model_change(self, selected_friendly_name):
        model_id = AVAILABLE_MODELS[selected_friendly_name]
        
        self.btn_encode.configure(state="disabled")
        self.btn_decode.configure(state="disabled")
        self.update_status(f"Loading {model_id}... Interface will be ready soon.")
        
        def load_task():
            # Cleanup previous model
            if self.stego:
                del self.stego
                gc.collect()
            
            try:
                # Use callback to pipe logs to GUI status bar
                self.stego = SafeNeuralStego(model_id, log_callback=lambda m: self.status_bar.configure(text=m))
                self.current_model_id = model_id
                
                # Enable UI
                self.after(0, lambda: self.btn_encode.configure(state="normal"))
                self.after(0, lambda: self.btn_decode.configure(state="normal"))
                self.after(0, lambda: self.update_status(f"Ready: {selected_friendly_name}"))
            except Exception as e:
                self.after(0, lambda: self.update_status(f"Error loading model: {e}"))

        threading.Thread(target=load_task, daemon=True).start()

    def run_encode(self):
        secret = self.entry_secret.get()
        start = self.entry_start_enc.get()
        if not secret: return
        
        self.btn_encode.configure(state="disabled")
        self.txt_out_enc.delete("0.0", "end")
        
        def task():
            try:
                res = self.stego.encode(secret.encode('utf-8'), start)
                self.after(0, lambda: self.txt_out_enc.insert("0.0", res))
                self.after(0, lambda: self.update_status("Encoding Complete."))
            except Exception as e:
                self.after(0, lambda: self.update_status(f"Error: {e}"))
            finally:
                self.after(0, lambda: self.btn_encode.configure(state="normal"))
        
        threading.Thread(target=task, daemon=True).start()

    def copy_enc(self):
        text = self.txt_out_enc.get("0.0", "end").strip()
        pyperclip.copy(text)
        self.update_status("Copied to clipboard.")

    def run_decode(self):
        text = self.txt_in_dec.get("0.0", "end").strip()
        start = self.entry_start_dec.get()
        if not text: return

        self.btn_decode.configure(state="disabled")
        self.txt_out_dec.delete("0.0", "end")

        def task():
            try:
                res = self.stego.decode(text, start)
                decoded_str = res.decode('utf-8', errors='replace')
                self.after(0, lambda: self.txt_out_dec.insert("0.0", decoded_str))
                self.after(0, lambda: self.update_status("Secret Revealed."))
            except Exception as e:
                self.after(0, lambda: self.txt_out_dec.insert("0.0", f"FAILURE: {e}"))
                self.after(0, lambda: self.update_status("Decoding Failed."))
            finally:
                self.after(0, lambda: self.btn_decode.configure(state="normal"))

        threading.Thread(target=task, daemon=True).start()

if __name__ == "__main__":
    app = NeuroStegoApp()
    app.mainloop()
