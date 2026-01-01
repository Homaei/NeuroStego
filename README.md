<div align="center">

# 🧠 NeuroStego
### Cryptographically Secure Linguistic Steganography

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-GPT2-green)](https://huggingface.co/gpt2)
[![Security](https://img.shields.io/badge/Security-AES256--GCM-red)](https://en.wikipedia.org/wiki/Galois/Counter_Mode)

**Hide your secrets inside innocent-looking, AI-generated text.**
<br>
*No metadata leakage. No statistical anomalies. Just natural language.*

[View Demo](#-demo) • [Installation](#-installation) • [The Science](#-mathematical-foundation) • [GUI & CLI](#-gui--cli-usage)

</div>

---

## 🕵️‍♂️ The Problem
Standard encryption (PGP, AES) protects the **content** of your message but leaks **metadata**.
When you send an encrypted blob like `U2FsdGVkX1+...`, any observer (ISP, firewall, network admin) knows *something* hidden is being transmitted.

## 🛡️ The Solution
**NeuroStego** transforms your encrypted bits into valid, context-aware natural language. It utilizes **Arithmetic Coding** over the probability distribution of a frozen GPT-2 model to perfectly map binary data to linguistic choices.

> **Alice sends:** "The stock market has shown resilience today..."
> **Bob decodes:** `CONFIDENTIAL_API_KEY_99X`

---

## ⚡ Demo

Try it yourself. The paragraph below is **NOT** random. It contains a hidden payload.

> **Cover Text:**
> "The weather today is significantly better than yesterday. We can expect clear skies and a gentle breeze throughout the afternoon. It is a perfect opportunity for outdoor activities or a long walk in the park."

If you run this text through the decoder, you will get: `Attack at Dawn!`

---

## 📐 Mathematical Foundation

NeuroStego is not a simple substitution cipher. It implements **Distribution-Transforming Steganography**.

We treat the Language Model (LLM) as a probability distribution function $P$.
Given a context $C$ (previous words), the model predicts the next token $w_t$:

$$P(w_t | w_0, ..., w_{t-1})$$

We map the interval $[0, 1)$ to the cumulative distribution of possible tokens. Your secret message (interpreted as a binary fraction $S \in [0, 1)$) determines which token is selected.

### Addressing Floating-Point Determinism
A critical flaw in most Neural Steganography implementations is **Floating Point Non-Determinism**. A probability calculated as `0.00300001` on an Intel CPU might be `0.00299999` on an AMD CPU or Apple Silicon, causing the decoding to fail completely.

**NeuroStego Solution:**
We implement a strictly deterministic sorting algorithm for candidate selection:
1.  Truncate probabilities to fixed precision.
2.  Sort candidates primarily by Probability (Desc), and secondarily by Token ID (Asc).
3.  Use **Power-of-2 Truncation** logic to ensure zero statistical bias in bit embedding.

---

## 🚀 Installation

Requires Python 3.8+ and PyTorch.

```bash
# Clone the repository
git clone https://github.com/Homaei/NeuroStego.git
cd NeuroStego

# Install dependencies
pip install -r requirements.txt
```

---

## �️ GUI & CLI Usage

We provide three ways to interact with NeuroStego:

### 1. Graphical User Interface (GUI)
A modern, dark-themed interface built with `CustomTkinter`.

```bash
python gui.py
```
*Features:*
*   Multi-Model Selector (GPT-2, DistilGPT, etc.)
*   Auto Copy-to-Clipboard
*   Status Bar Logging

### 2. Command Line Interface (CLI)
Interactive mode for servers or headless environments.

```bash
python cli.py
```

### 3. Google Colab (Standalone)
If you want to run this in a notebook (e.g., Google Colab) without cloning the whole repo, use `colab_standalone.py` or copy its content. It auto-installs dependencies.

---

## 📖 Library Usage (Alice & Bob Scenario)

### Scenario
Alice wants to send a secret password to Bob. They both have this repository installed.

### Step 1: Alice Encodes the Secret
Alice runs the script to hide her secret message inside a text about "The future of AI".

```python
from stego import SafeNeuralStego

# Initialize the secure engine
stego = SafeNeuralStego(model_name='gpt2')

# The secret payload
secret = b"MyPassword123"

# Encode it into innocent text
cover_text = stego.encode(secret, start_text="The future of AI is")

print(cover_text)
# Output: "The future of AI is promising because deep learning..."
```

### Step 2: Transmission
Alice sends the output text to Bob via Email, WhatsApp, or LinkedIn. To an observer, it looks like a normal opinion on Tech.

### Step 3: Bob Decodes the Message
Bob copies the text he received and runs the decoder. He does not need to know the length; the protocol handles it via a 32-bit header.

```python
from stego import SafeNeuralStego

# Initialize (must match sender's model)
stego = SafeNeuralStego(model_name='gpt2')

# Bob receives the text
received_text = "The future of AI is promising because deep learning..."

# Decode
recovered_payload = stego.decode(received_text)

print(recovered_payload)
# Output: b"MyPassword123"
```

---

## ⚠️ Scientific Limitations & Security Notice
1. **Model Synchronization:** Both Sender and Receiver MUST use the exact same model version (e.g., `gpt2`). If HuggingFace updates the weights, decoding will fail.
2. **Context Dependence:** The generated text relies on the previous tokens. You cannot modify even a single comma of the cover_text during transmission, or the entire chain breaks (Avalanche Effect).
3. **Efficiency:** The embedding rate is approximately 1-3 bits per word depending on the entropy of the sentence. A 1KB secret might result in a 3KB-5KB text.

---

## 📜 License
Distributed under the MIT License.
