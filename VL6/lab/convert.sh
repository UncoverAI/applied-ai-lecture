#!/bin/bash

%cd /kaggle/working
!git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
%cd /kaggle/working/llama.cpp
!sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
!LLAMA_CUDA=1 conda run -n base make -j > /dev/null

!python /kaggle/working/llama.cpp/convert_hf_to_gguf.py /kaggle/input/merging-and-exporting-fine-tuned-llama-3-2/llama-3.2-3b-it-Ecommerce-ChatBot \
    --outfile /kaggle/working/llama-3.2-3b-ecommerce-chatbot.gguf \
    --outtype f16 \
    --model llama-3.2
    
%cd /kaggle/working
!./llama.cpp/llama-quantize /kaggle/working/llama-3.2-3b-ecommerce-chatbot.gguf /kaggle/working/llama-3.2-3b-ecommerce-chatbot-Q4_K_M.gguf Q4_K_M

from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/kaggle/working/llama-3.2-3b-ecommerce-chatbot-Q4_K_M.gguf",
    path_in_repo="llama-3.2-3b-ecommerce-chatbot-Q4_K_M.gguf",
    repo_id="kingabzpro/llama-3.2-3b-it-Ecommerce-ChatBot",
    repo_type="model",
)