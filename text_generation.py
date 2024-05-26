import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, model_name='gpt2', max_length=200, num_return_sequences=1):
    # トークナイザーとモデルのロード
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # モデルを評価モードに設定
    model.eval()

    # プロンプトをトークン化
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # テキストの生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # サンプリングを有効にすることで多様な生成が可能
            top_k=50,        # トップKサンプリングを有効に
            top_p=0.95       # トップPサンプリングを有効に
        )

    # 生成されたテキストのデコード
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_texts = generate_text(prompt, max_length=100, num_return_sequences=1)

    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i + 1}:\n{text}\n")
