from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def nllb():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    #model = model.to(device)
    # Example English text
    # article = "yo soy un hombre"

    # Tokenize the text
    # inputs = tokenizer(article, return_tensors="pt")

    # Generate translation using the model
    # translated_tokens = model.generate(
    #     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
    # )

    # Decode and print the translated text
    #print(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])
    return model, tokenizer

def nllb_translate(model, tokenizer, article, language):
    # Tokenize the text
    inputs = tokenizer(article, return_tensors="pt")

    # Move the tokenized inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    if language == "spanish":
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"], max_length=30
        )
    elif language == "english":
        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30
        )
    

    text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return text

