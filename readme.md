---
license: agpl-3.0
base_model:
- meta-llama/Llama-3.1-8B-Instruct
---

# 🧠 Solphie-1S Foundation Model  

CA:H62qVjizU7vNtqVJVAiy2A1541atw856KPtAsLyGpump

![Solphie](https://i.ibb.co/bjCbhFJ5/solphie-banner.png)  

[![License](https://img.shields.io/badge/license-AGPL%20v3-blue?style=flat-square)](https://www.gnu.org/licenses/agpl-3.0.html)  
[![HF](https://img.shields.io/badge/HuggingFace-Solphie--1S--Foundation--Model-orange?style=flat-square&logo=huggingface)](https://huggingface.co/Virende/Solphie-1S-Foundation-Model)  

---

## 1️⃣ **Overview**  

The **Solphie-1S-Foundation-Model** is a fine-tuned adaptation of Meta's LLaMA 3.1 8B model, designed to deliver **precision-driven AI intelligence** for **Solana developers**. This model seamlessly integrates **on-chain data, fine-tuned instruction sets, and multi-turn reasoning** to optimize dApp development, smart contract interaction, and blockchain intelligence.  

### ✅ **Key Capabilities**  
- **Answer complex Solana-related queries**  
- **Generate high-quality, Solana-optimized code snippets**  
- **Debug smart contracts and dApps**  
- **Explain technical blockchain concepts with clarity and depth**  

This model is engineered to bridge **AI intelligence and blockchain development**, empowering developers to build, optimize, and scale with **on-chain knowledge** at their fingertips.  

**(Knowledge cut-off date: January 29, 2025)**  

---

## 2️⃣ **Key Features**  

- Fine-tuned with **developer-first instruction tuning**, optimized for Solana workflows.  
- Efficient and lightweight via **LoRA (Low-Rank Adaptation)**, ensuring scalable fine-tuning.  
- **Retains context across multi-turn conversations**, enabling seamless AI-assisted development.  
- Generates **complete, executable code snippets** with real-world applications.  

---

## 3️⃣ **Model Card**  

| **Parameter**             | **Details**                                                    |
|---------------------------|----------------------------------------------------------------|
| **Base Model**            | Meta LLaMa 3.1 8B                                              |
| **Fine-Tuning Framework** | Hugging Face Transformers, LoRA                               |
| **Dataset Size**          | 13,593 high-quality Q&A pairs                                |
| **Context Length**        | 4,096 tokens                                                 |
| **Training Steps**        | 10,000                                                       |
| **Learning Rate**         | 3e-4                                                         |
| **Batch Size**            | 1 per GPU with gradient accumulation                         |
| **Epochs**                | 2                                                            |
| **Model Size**            | 8 billion parameters (adapter size ~10 MB)                  |
| **Pre-trained Tasks**     | Instruction following, Code generation, Debugging, Multi-turn Q&A |

---

## 4️⃣ **Installation and Usage**  

### 4.1 **Install Dependencies**  

```bash
pip install transformers datasets peft wandb
```

### 4.2 **Load the Model**  

```python
from transformers import LlamaForCausalLM, AutoTokenizer

model_name = "Virende/Solphie-1S-Foundation-Model"

model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 4.3 **Run Inference**  

```python
def complete_chat(model, tokenizer, messages, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

response = complete_chat(model, tokenizer, [
    {"role": "system", "content": "You are Solphie, an AI expert in Solana development."},
    {"role": "user", "content": "Explain how to interact with Raydium API for token swaps."}
])
print(response)
```

---

## 5️⃣ **Dataset**  

| Split   | Count  | Description                    |
|---------|--------|--------------------------------|
| **Train** | 27.1k | High-quality Q&A pairs        |

### **5.1 Dataset Format (JSONL)**  

```json
{
  "question": "How to use the Helius API for transaction indexing?",
  "answer": "To index transactions, use Helius's Webhooks API ...",
  "chunk": "Helius API allows you to set up ..."
}
```

### **5.2 Download Dataset**  

```python
from datasets import load_dataset
ds = load_dataset("Virende/Solphie-1S-Foundation-Model-DS")
```

---

## 6️⃣ **Model Training & Fine-tuning**  

### 6.1 **Download Model**  

```bash
git lfs install
git clone https://huggingface.co/Virende/Solphie-1S-Foundation-Model
```

### 6.2 **Train the Model**  

```bash
python3 train.py
```

### 6.3 **Run Demo**  

```bash
python3 demo.py
```

---

## 7️⃣ **Technical Insights**  

### **7.1 LoRA Configuration**  

- Rank: 8  
- Alpha: 32  
- Dropout: 0.01  
- Adapter Size: ~10 MB  

### **7.2 Optimization Techniques**  

- **Mixed Precision (FP16)** for faster inference.  
- **Gradient Accumulation** for memory efficiency.  
- **Parameter-efficient fine-tuning** to preserve base model knowledge.  

---

## 8️⃣ **Community & Support**  

For discussions, updates, and support:  
- **Twitter**: [@SolphieAI](https://x.com/SolphieAI)  
- **Hugging Face**: [Model Hub](https://huggingface.co/Solphie/Solphie-1S-Foundation-Model)  

---

## 9️⃣ **License**  

This model is released under the **GNU Affero General Public License v3.0 (AGPLv3)**.  

---

## 🔟 **Acknowledgments**  

Special thanks to the **Solana ecosystem developers, the Web3 AI community, and open-source contributors** for their invaluable support and contributions.  

---

## 🔗 **Related Resources**  

- [Solana Official Documentation](https://docs.solana.com)  
- [Hugging Face Model Repo](https://huggingface.co/Solphie/Solphie-1S-Foundation-Model)  
- [Solana Developer Tools](https://solana.com/developers)  
```

---
