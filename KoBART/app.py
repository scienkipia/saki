from transformers import PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer
from transformers import BartForConditionalGeneration
import streamlit as st
import torch



def tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    return tokenizer


@st.cache(allow_output_mutation=True)
def get_model():
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
    model.eval()
    return model


default_text = 'NULL'
with open('result.txt','r') as default_file:
  default_text = default_file.read()


model = get_model()
tokenizer = tokenizer()
st.title("Summarization Model Test")
text = st.text_area("Input news :", value=default_text)

st.markdown("## Original News Data")
st.write(text)

if text:
    st.markdown("## Predict Summary")
    with st.spinner('processing..'):
        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + \
            raw_input_ids + [tokenizer.eos_token_id]
        summary_ids = model.generate(torch.tensor([input_ids]),
                                     max_length=256,
                                     early_stopping=True,
                                     repetition_penalty=2.0)
        summ = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    with open('summarized.txt','w') as fResult:
      fResult.write(summ)

