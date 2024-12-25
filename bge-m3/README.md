
---
tasks:
- embedding
license: Apache License 2.0
---
## bge-m3

This repo is a mirror of embedding model bge-m3.

## Information
- dimensions: 1024
- max_tokens: 8192
- language: zh, en

## Example code

### Install packages
```bash
pip install xinference[ggml]>=0.4.3
```
If you want to run with GPU acceleration, refer to [installation](https://github.com/xorbitsai/inference#installation).

###  Start a local instance of Xinference
```bash
xinference -p 9997
```

### Launch and inference
```python
from xinference.client import Client

client = Client("http://localhost:9997")
model_uid = client.launch_model(
    model_name="bge-m3",
    model_type="embedding"
    )
model = client.get_model(model_uid)

input_text = "What is the capital of China?"
model.create_embedding(input_text)
```

### More information

[Xinference](https://github.com/xorbitsai/inference) Replace OpenAI GPT with another LLM in your app 
by changing a single line of code. Xinference gives you the freedom to use any LLM you need. 
With Xinference, you are empowered to run inference with any open-source language models, 
speech recognition models, and multimodal models, whether in the cloud, on-premises, or even on your laptop.

<i><a href="https://join.slack.com/t/xorbitsio/shared_invite/zt-1z3zsm9ep-87yI9YZ_B79HLB2ccTq4WA">👉 Join our Slack community!</a></i>


