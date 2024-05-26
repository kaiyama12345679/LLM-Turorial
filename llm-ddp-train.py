from datetime import timedelta, datetime
import torch
import torch.distributed as dist
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
torch.multiprocessing.set_start_method("spawn")
from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed
import random
import numpy as np
import os
from dotenv import load_dotenv
import wandb
import time
from discord_webhook import send_discord_webhook

load_dotenv()
if os.getenv("WANDB_API_KEY") is None:
    raise ValueError("API key for wandb is not set")
wandb.login()

MODEL_NAME = "gpt2"
PROMPT = "Once upon a time"
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_dataloder(tokenizer, batch_size=4):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize_function_with_labels(examples):                            
        inputs = tokenizer(                                                 # tokenization for text, set 'input_ids' and 'labels'
            examples['text'], 
            padding     = 'max_length', 
            truncation  = True, 
            max_length  = 512
        )
        inputs["labels"] = inputs["input_ids"].copy()                       # 'input_ids' as 'labels'
        return inputs
    
    dataset = dataset.map(tokenize_function_with_labels, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = dataset["train"]
    print("Train dataset length: ", len(train_dataset))
    eval_dataset = dataset["validation"]
    print("Eval dataset length: ", len(eval_dataset))
    test_dataset = dataset["test"]
    print("Test dataset length: ", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, eval_dataloader, test_dataloader

def set_model(tokenizer):
    config = GPT2Config.from_pretrained(MODEL_NAME, output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, config=config)
    return model


def set_optimizer_scheduler(model, total_steps, learning_rate, warmup_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    return optimizer, scheduler


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(timedelta(seconds=elapsed_rounded))
def train_ddp(batch_size, epochs, learning_rate, warmup_steps, run_name):
    # 現在の日時を取得
    now = datetime.now()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 時間と分を取得
    hour = now.hour
    minute = now.minute

    # hhmm形式にフォーマット
    hhmm = "{:02d}{:02d}".format(hour, minute)

    save_path = f"./output/{run_name}-{hhmm}"
    set_seed(42)
    try:

        deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1, offload_param_device='cpu')
        accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, log_with="wandb")
        accelerator.init_trackers(
            project_name="llm-train-tutorial",
            init_kwargs={"wandb": {"group": run_name, "name": MODEL_NAME + hhmm}},
            config={"model_name": MODEL_NAME,"batch_size": batch_size, "epochs": epochs, "learning_rate": learning_rate, "warmup_steps": warmup_steps},
        )
        table = []
        if accelerator.is_main_process:
            wandb_run = accelerator.get_tracker("wandb")
            send_discord_webhook("Training Started")
        train_data_loader, eval_data_loader, test_data_loader = set_dataloder(tokenizer, batch_size=batch_size)
        model = set_model(tokenizer)

        total_steps = len(train_data_loader) * epochs
        optimizer, scheduler = set_optimizer_scheduler(model, total_steps, learning_rate, warmup_steps)

        model, optimizer, train_data_loader, eval_data_loader, test_data_loader, scheduler = accelerator.prepare(
            model, optimizer, train_data_loader, eval_data_loader, test_data_loader, scheduler
        )
        for epoch_i in range(epochs):
            if accelerator.is_main_process:
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')

            t0 = time.time()
            total_loss = 0
            model.train()

            for step, batch in enumerate(train_data_loader):
                model.train()
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % 300 == 0 and step != 0:
                    if accelerator.is_main_process:
                        t1 = time.time()
                        time_elapsed = format_time(t1 - t0)
                        print(f"  Batch {step} of {len(train_data_loader)} took: {time_elapsed}")

                        model.eval()
                        global_step = len(train_data_loader) * epoch_i + step
                        with torch.no_grad():
                            tokenized_prompt = tokenizer(PROMPT, return_tensors="pt")
                            tokenized_prompt = {k: v.to("cuda") for k, v in tokenized_prompt.items()}
                            sample_outputs = accelerator.unwrap_model(model).generate(
                                **tokenized_prompt,
                                max_length=512,
                                num_return_sequences=1,
                                pad_token_id=tokenizer.eos_token_id,
                                temperature=1.0,
                                top_k=50,
                                do_sample=True,
                                top_p=0.95,
                            )
                            generated_text = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
                            print(generated_text[0])
                            table.append([epoch_i, global_step, generated_text[0]])
                            wandb_run.log_table(table_name='generated_text', columns=["epoch", "global_step", "text"], data=table)


                if step % 500 == 0 and step != 0:
                    model.eval()
                    total_eval_loss = 0
                    for batch in eval_data_loader:
                        batch = {k: v.to("cuda") for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        total_eval_loss += loss.item()
                    avg_eval_loss = total_eval_loss / len(eval_data_loader)
                    if accelerator.is_main_process:
                        print(f"  Evaluation Loss: {avg_eval_loss}")
                        accelerator.log({"eval_loss": avg_eval_loss})
                    model.train()
            avg_train_loss = total_loss / len(train_data_loader)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print("")
                print(f"Average training loss: {avg_train_loss}")
                accelerator.log({"train_loss": avg_train_loss})
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_path, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("")
            print("Training complete!")
            if not os.path.exists("./output/"):
                os.makedirs("./output/")
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_path, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))

            print("Saved model to", save_path)
            print("Runnning evaluation on test dataset")
        total_test_loss = 0
        model.eval()
        for batch in test_data_loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_data_loader)
        if accelerator.is_main_process:
            print(f"Test Loss: {avg_test_loss}")
            accelerator.log({"test_loss": avg_test_loss})
        
    except Exception as e:
        if accelerator.is_main_process:
            send_discord_webhook(f"Error Occured!:  {str(e)}")
        raise(e)
    else:
        if accelerator.is_main_process:
            send_discord_webhook("Training Finished")
    finally:
        accelerator.end_training()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--warmup_steps", type=int, default=1e4, help="Warmup steps for training")
    parser.add_argument("--run_name", type=str, default="llm_train_group", help="Run name for wandb")
    args = parser.parse_args()
    train_ddp(args.batch_size, args.epochs, args.learning_rate, args.warmup_steps, args.run_name)
    

        

            
                    


                    




