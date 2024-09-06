import torch
import torch.nn as nn
from NN.model_predict_hand import *
from transformers import GPT2Model, GPT2Config
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm


path = ouput_path = os.getcwd() + "/../../../dataset/output"
save_directory = os.getcwd() + "/checkpoints"
os.makedirs(save_directory, exist_ok=True)

file_list = os.listdir(path)

# 从文件列表中找出以.gz结尾的文件并去除后缀
files = []
for file in file_list:
    if file.endswith(".npz"):
        files.append(file[:-10])

train_dataset = MahjongDataset(path, files[:100])
eval_dataset = MahjongDataset(path, files[100:110])


accelerator = Accelerator()
accelerator.state.ddp_kwargs = {"find_unused_parameters": True}

device = accelerator.device
config = GPT2Config(n_positions=256, n_embd=512, n_layer=12, n_head=16)
my_model = myModel(config)
my_model.to(device)
my_model.to(torch.float32)


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=MahjongCollator())
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=MahjongCollator())
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4)



train_dataloader, eval_dataloader, my_model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, my_model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

train_progress_bar = tqdm(range(num_training_steps))
eval_progress_bar = tqdm(range(len(eval_dataloader)))

my_model.train()
for epoch in range(num_epochs):
    my_model.train()
    for step, batch in enumerate(train_dataloader):
        output = my_model(batch)
        if output == None:
            print("output is None")
            optimizer.zero_grad()
            continue
        loss = output['loss']
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        train_progress_bar.update(1)
        if step % 2000 == 0:
            # 保存checkpoint
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(my_model)
            accelerator.save(unwrapped_model.state_dict(), os.path.join(save_directory, f"checkpoint_{epoch}_{step}.pt"))

    my_model.eval()
    eval_losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            output = my_model(batch)
        if output is None:
            print("output is None")
            continue
        loss = output['loss']
        eval_losses.append(loss.item())
        eval_progress_bar.update(1)

    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    print(f"Epoch {epoch} Evaluation Loss: {avg_eval_loss}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(my_model)
    accelerator.save(unwrapped_model.state_dict(), os.path.join(save_directory, f"final_checkpoint_epoch_{epoch}.pt"))


    # print(f"epoch {epoch} done")