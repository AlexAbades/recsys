import torch 
import json 

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print("Saving checkpoint to", chk_path)
    torch.save(
        {
            "epoch": epoch + 1,
            "lr": lr,
            "optimizer": optimizer.state_dict(),
            "model_pos": model_pos.state_dict(),
            "min_loss": min_loss,
        },
        chk_path,
    )

def calculate_model_size(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_bytes = trainable_params * 4  # for 32-bit floats
    model_size_mb = model_size_bytes / (1024**2)
    print(f"INFO: Model Size: {model_size_mb:.2f} MB")
    print("INFO: Trainable parameter count:", trainable_params)


def calculate_memory_allocation():
    memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # Convert to MB
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"INFO: Memory Allocated: {memory_allocated:.2f} MB")
    print(f"INFO: Peak Memory Allocated: {peak_memory_allocated:.2f} MB")

def save_accuracy(chk_path:str, **kwargs):
    print("Saving acuracy to", chk_path +'_acuracy.json')
    with open(chk_path + '_acuracy.json', 'w') as file:
        json.dump(kwargs, file)