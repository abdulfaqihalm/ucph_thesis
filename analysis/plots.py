from matplotlib import pyplot as plt 
import pandas as pd

def plot_loss_function(result_path:str, output_path:str, output_name:str) -> None:
    
    data = pd.read_csv(result_path)
    
    plt.plot(data["epoch"], data["train_loss"], label="Train Loss", color="blue")
    plt.plot(data["epoch"], data["val_loss"], label="Validation Loss", color="orange", linestyle="dashed")
    plt.title("Loss Function")
    plt.xlabel("Number of Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_path}/{output_name}.png")

if __name__ == "__main__":
    result_path = "data/outputs/training_20240306_1202.log"
    output_path = "data/outputs/analysis"
    output_name = "loss_funciton"

    plot_loss_function(result_path, output_path, output_name) 