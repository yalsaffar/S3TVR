from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def nllb():
    """
    Load and return the NLLB (No Language Left Behind) model and tokenizer.

    This function loads the NLLB-200-distilled-1.3B model and tokenizer from Hugging Face's Transformers library.
    The model is configured to use a GPU if available, otherwise it defaults to CPU.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
            - model (transformers.AutoModelForSeq2SeqLM): The loaded NLLB model.
            - tokenizer (transformers.AutoTokenizer): The loaded tokenizer.
            
    Example usage:
        model, tokenizer = nllb()
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
    return model, tokenizer

def nllb_translate(model, tokenizer, article, language):
    """
    Translate an article using the NLLB model and tokenizer.

    Args:
        model (transformers.AutoModelForSeq2SeqLM): The NLLB model to use for translation.
            Example: model, tokenizer = nllb()
        tokenizer (transformers.AutoTokenizer): The tokenizer to use with the NLLB model.
            Example: model, tokenizer = nllb()
        article (str): The article text to be translated.
            Example: "This is a sample article."
        language (str): The target language for translation. Must be either 'spanish' or 'english'.
            Example: "spanish"

    Returns:
        str: The translated text.
            Example: "Este es un artículo de muestra."
    """
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

