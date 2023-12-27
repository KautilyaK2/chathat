from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import streamlit as st
# Choose any model available at https://health.petals.dev
model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
system_prompt = "### System:\nYou are Stable Beluga, an AI that is very precise. Be as accurate as you can.\n\n"
conv=system_prompt
message = st.chat_input('Message')
if message:
    prompt = f"### User: {message}\n\n### Assistant:\n"
    conv+=prompt
    # Run the model as if it were on your computer
    inputs = tokenizer(conv, return_tensors="pt")["input_ids"]
    outputs = model.generate(inputs, max_new_tokens=256)
    with st.chat_message('user'):
        st.write(tokenizer.decode(outputs[0])[3:-4])
