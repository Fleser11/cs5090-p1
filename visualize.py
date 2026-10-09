
import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import argparse

def main(
    input: str,
    output: str,
    title: str
    ):

    os.makedirs("/".join(input.split("/")[:-1]), exist_ok=True)
    os.makedirs("/".join(output.split("/")[:-1]), exist_ok=True)

    train_iters = []
    train_loss_arr = []
    val_iters = []
    val_loss_arr = []
    with open(input, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:

            # if line["train_loss"] != "":
            if line["val_loss"] != "":
                val_loss_arr.append(float(line["val_loss"]))
                val_iters.append(float(line["iteration"]))
                train_loss_arr.append(float(line["train_loss"]))
                train_iters.append(float(line["iteration"]))


    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(np.array(train_iters), np.array(train_loss_arr), label="Training Loss")
    axs.plot(np.array(val_iters), np.array(val_loss_arr), label="Validation Loss")
    axs.set(
        xlabel="Iterations",
        ylabel="Loss",
        title=title
    )
    plt.legend()
    # fig.set_label(("Iterations", "Loss"))

    plt.savefig(output, dpi=300)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input", required=True
    )
    arg_parser.add_argument(
        "--output", required=True
    )
    arg_parser.add_argument(
        "--title", required=True
    )
    args = arg_parser.parse_args()
    main(args.input, args.output, args.title)

