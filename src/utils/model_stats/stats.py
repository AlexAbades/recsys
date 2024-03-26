import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def save_model_with_params(chk_path, model, model_params):
    print("Saving checkpoint to", chk_path)
    torch.save(
        {"model_state_dict": model.state_dict(), "model_params": model_params}, chk_path
    )


def save_for_finetuning(chk_path, model, optimizer=None, additional_info={}):
    save_content = {
        "model_state_dict": model.state_dict(),
        "additional_info": additional_info,
    }
    if optimizer:
        save_content["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(save_content, chk_path)


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


def json_default_serializer(obj):
    """Custom JSON serializer for unsupported data types."""
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_accuracy(chk_path: str, **kwargs):
    print("Saving accuracy to", chk_path + "_accuracy.json")
    with open(chk_path + "_accuracy.json", "w") as file:
        # Use the default parameter to specify the custom serializer
        json.dump(kwargs, file, default=json_default_serializer)


def save_model_specs(model, folder_path, filename="model_specs.txt"):
    """
    Saves the specifications of a PyTorch model to a file.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose specifications are to be saved.
        folder_path (str): The path to the folder where the specifications file should be saved.
        filename (str): The name of the file to save the specifications to. Default is 'model_specs.txt'.
    """

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Construct the full path for the file
    file_path = os.path.join(folder_path, filename)

    # Open the file at the specified path and write the model's specifications
    with open(file_path, "w") as f:
        print(model, file=f)

    print(f"Model specifications saved to {file_path}")


def plot_and_save_losses(loss_dict, folder_path, filename="loss_plot.png"):
    """
    Plots the training losses and saves the plot to a given folder path.

    Parameters:
        loss_dict (dict): A dictionary where keys are epoch numbers and values are loss values.
        folder_path (str): The path to the folder where the plot should be saved.
        filename (str): The name of the file to save the plot as. Default is 'loss_plot.png'.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Sort the dictionary by epoch (key) to ensure the plot is in order
    epochs = sorted(loss_dict.keys())
    losses = [
        (
            loss_dict[epoch].cpu().item()
            if torch.is_tensor(loss_dict[epoch])
            else loss_dict[epoch]
        )
        for epoch in epochs
    ]

    # Filter out infinite values and find the maximum loss for finite values
    finite_losses = [loss for loss in losses if loss != float("inf")]
    max_loss = max(finite_losses) if finite_losses else 0

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker="o", linestyle="-", color="blue")
    plt.title("Test Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # plt.ylim(0, max_loss)

    plt.tight_layout()  # Adjust the layout to make room for the labels

    # Save the plot
    plot_path = os.path.join(folder_path, filename)
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


def save_opts_args(opts, args, folder_path, filename="opts_args.txt"):
    """
    Saves the command-line options and arguments to a file.

    Parameters:
        opts: Command-line options object (e.g., from argparse).
        args: List of command-line arguments.
        folder_path (str): Path to the folder where the file should be saved.
        filename (str): Name of the file to save the options and arguments to.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Full path for the file
    file_path = os.path.join(folder_path, filename)

    with open(file_path, "w") as f:
        # Write options
        f.write("Options:\n")
        for opt, value in vars(opts).items():
            f.write(f"{opt}: {value}\n")

        # Write arguments
        f.write("\nArguments:\n")
        for arg in args:
            f.write(f"{arg}\n")

    print(f"Options and arguments saved to {file_path}")


def save_dict_to_file(dict_data, folder_path, filename="dict_contents.txt"):
    """
    Saves the contents of an `easydict` dictionary to a text file in a readable format.

    Parameters:
        dict_data (edict): The `easydict` dictionary instance to be saved.
        folder_path (str): The path to the folder where the file should be saved.
        filename (str): The name of the file to save the dictionary contents to. Default is 'dict_contents.txt'.
    """

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Construct the full path for the file
    file_path = os.path.join(folder_path, filename)

    # Serialize the easydict to a JSON string for a readable format
    dict_str = json.dumps(dict_data, indent=4)

    # Open the file and write the serialized string
    with open(file_path, "w") as f:
        f.write(dict_str)

    print(f"Dictionary contents saved to {file_path}")
