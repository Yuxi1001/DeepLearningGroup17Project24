from mortality_part_preprocessing import load_pad_separate
from mortality_classification import train_test
import os
import click
import torch
import random
import numpy as np
import json
# from models.EHRMamba2 import Mamba2Pretrain
# from models.EHRMamba import MambaPretrain, MambaFinetune



@click.command()
@click.option(
    "--output_path",
    default="./ehr_classification_results/",
    help="Path to output folder",
)
@click.option("--pooling", default="max", help="pooling function")
@click.option("--epochs", default=300, help="model dropout rate")
@click.option("--dropout", default=0.4, help="model dropout rate")
@click.option("--attn_dropout", default=0.4, help="model attention dropout rate")
@click.option(
    "--model_type", default="transformer", help="model_type"
)
@click.option("--heads", default=1, help="number of attention heads")
@click.option("--batch_size", default=64, help="batch size")
@click.option("--layers", default=1, help="number of attention layers")
@click.option("--dataset_id", default="physionet2012", help="filename id of dataset")
@click.option("--base_path", default="./P12data", help="Path to data folder")
@click.option("--lr", default=0.001, help="learning rate")
@click.option("--patience", default=10, help="patience for early stopping")
@click.option(
    "--use_mask",
    default=False,
    help="boolean, use mask for timepoints with no measurements",
)
@click.option(
    "--early_stop_criteria",
    default="auroc",
    help="what to early stop on. Options are: auroc, auprc, auprc+auroc, or loss",
)
@click.option("--seft_n_phi_layers", default=3)
@click.option("--seft_phi_width", default=32)
@click.option("--seft_phi_dropout", default=0.)
@click.option("--seft_n_psi_layers", default=2)
@click.option("--seft_psi_width", default=64)
@click.option("--seft_psi_latent_width", default=128)
@click.option("--seft_dot_prod_dim", default=128)
@click.option("--seft_latent_width", default=128)
@click.option("--seft_n_rho_layers", default=3)
@click.option("--seft_rho_width", default=32)
@click.option("--seft_rho_dropout", default=0.)
@click.option("--seft_max_timescales", default=100)
@click.option("--seft_n_positional_dims", default=4)
@click.option("--ipnets_imputation_stepsize", default=0.25)
@click.option("--ipnets_reconst_fraction", default=0.25)
@click.option("--recurrent_dropout", default=0.3)
@click.option("--recurrent_n_units", default=100)
# # New added for Mamba2
# @click.option("--embedding_size", default=768, help="Size of embeddings")
# @click.option("--max_num_visits", default=512, help="Maximum number of visits")
# @click.option("--max_seq_length", default=2048, help="Maximum sequence length")
# @click.option("--state_size", default=64, help="State size")
# @click.option("--num_heads", default=24, help="Number of attention heads")
# @click.option("--head_dim", default=64, help="Head dimension size")
# @click.option("--num_hidden_layers", default=32, help="Number of hidden layers")
# @click.option("--expand", default=2, help="Expand factor")
# @click.option("--conv_kernel", default=4, help="Convolution kernel size")
# @click.option("--dropout_prob", default=0.1, help="Dropout probability")
# @click.option("--padding_idx", default=0, help="Padding index")
# @click.option("--cls_idx", default=1, help="CLS token index")
# @click.option("--eos_idx", default=2, help="EOS token index")
# @click.option("--n_groups", default=1, help="Number of groups")
# @click.option("--chunk_size", default=256, help="Chunk size")
# New added for Mamba
@click.option("--embedding_size", default=768, help="Size of embeddings")
@click.option("--time_embeddings_size", default=32, help="Size of time embeddings")
@click.option("--visit_order_size", default=3, help="Size of visit order embeddings")
@click.option("--type_vocab_size", default=9, help="Type vocabulary size")
@click.option("--max_num_visits", default=512, help="Maximum number of visits")
@click.option("--max_seq_length", default=2048, help="Maximum sequence length")
@click.option("--state_size", default=16, help="State size")
@click.option("--num_hidden_layers", default=32, help="Number of hidden layers")
@click.option("--expand", default=2, help="Expand factor")
@click.option("--conv_kernel", default=4, help="Convolution kernel size")
@click.option("--learning_rate", default=5e-5, help="Learning rate")
@click.option("--dropout_prob", default=0.1, help="Dropout probability")
@click.option("--padding_idx", default=0, help="Padding index")
@click.option("--cls_idx", default=5, help="CLS token index")

def core_function(
    output_path,
    base_path,
    model_type,
    epochs,
    dataset_id,
    batch_size,
    lr,
    patience,
    early_stop_criteria,
    **kwargs
):

    model_args = kwargs

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    accum_loss = []
    accum_accuracy = []
    accum_auprc = []
    accum_auroc = []

    for split_index in range(1, 6):

        base_path_new = f"{base_path}/split_{split_index}"
        train_pair, val_data, test_data = load_pad_separate(
            dataset_id, base_path_new, split_index
        )
        # train_pair = [(d.long(), t, s, l) for (d, t, s, l) in train_pair]  # Convert to LongTensor
        # train_pair = [
        #     (d_pos.long(), t_pos, s_pos, l_pos, m_pos, del_pos)
        #     for ((d_pos, t_pos, s_pos, l_pos, m_pos, del_pos), _) in train_pair
        # ]
        # Process train_pair while maintaining the pairing structure
        # print("First element of train_pair:", train_pair[0])
        # print("Type of first element:", type(train_pair[0]))
        # print("Length of first element:", len(train_pair[0]))

        # train_pair = [
        #     (
        #         (d_pos.long(), t_pos, s_pos, l_pos, m_pos, del_pos),
        #         (d_neg.long(), t_neg, s_neg, l_neg, m_neg, del_neg)
        #     )
        #     for ((d_pos, t_pos, s_pos, l_pos, m_pos, del_pos),
        #         (d_neg, t_neg, s_neg, l_neg, m_neg, del_neg)) in train_pair
        # ]
        train_pair = [
            (
                (d_pos, t_pos, s_pos, l_pos, m_pos, del_pos),
                (d_neg, t_neg, s_neg, l_neg, m_neg, del_neg)
            )
            for ((d_pos, t_pos, s_pos, l_pos, m_pos, del_pos),
                (d_neg, t_neg, s_neg, l_neg, m_neg, del_neg)) in train_pair
        ]

        # val_data = [(d.long(), t, s, l) for (d, t, s, l) in val_data]
        # test_data = [(d.long(), t, s, l) for (d, t, s, l) in test_data]
        # val_data = [
        #     (d.long(), t, s, l, m, deltas)
        #     for (d, t, s, l, m, deltas) in val_data
        # ]
        # test_data = [
        #     (d.long(), t, s, l, m, deltas)
        #     for (d, t, s, l, m, deltas) in test_data
        # ]
        val_data = [
            (d, t, s, l, m, deltas)
            for (d, t, s, l, m, deltas) in val_data
        ]
        test_data = [
            (d, t, s, l, m, deltas)
            for (d, t, s, l, m, deltas) in test_data
        ]
       


        # make necessary folders
        # if new model, make model folder
        if os.path.exists(output_path):
            pass
        else:
            try:
                os.mkdir(output_path)
            except OSError as err:
                print("OS error:", err)
        # make run folder
        base_run_path = os.path.join(output_path, f"split_{split_index}")
        run_path = base_run_path
        if os.path.exists(run_path):
            pass
        else:
            try:
                os.mkdir(run_path)
            except OSError as err:
                print("OS error:", err)
                # raise ValueError(f"Path {run_path} already exists.")
        # if not os.path.exists(run_path):
        #     os.makedirs(run_path)

        # os.mkdir(run_path)

        # save model settings
        model_settings = {
            "model_type": model_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": dataset_id,
            "learning_rate": lr,
            "patience": patience,
            "early_stop_criteria": early_stop_criteria,
            "base_path": base_path,
            "pooling_fxn": model_args["pooling"],
        }
        if model_type == "transformer":
            model_settings["layers"] = model_args["layers"]
        if model_type in ("seft", "transformer"):
            model_settings["dropout"] = model_args["dropout"]
            model_settings["attn_dropout"] = model_args["attn_dropout"]
            model_settings["use_timepoint_mask"] = model_args["use_mask"]
            model_settings["heads"] = model_args["heads"]
        if model_type == "seft":
            model_settings["seft_n_phi_layers"] = model_args["seft_n_phi_layers"]
            model_settings["seft_phi_width"] = model_args["seft_phi_width"]
            model_settings["seft_phi_dropout"] = model_args["seft_phi_dropout"]
            model_settings["seft_n_psi_layers"] = model_args["seft_n_psi_layers"]
            model_settings["seft_psi_width"] = model_args["seft_psi_width"]
            model_settings["seft_psi_latent_width"] = model_args["seft_psi_latent_width"]
            model_settings["seft_dot_prod_dim"] = model_args["seft_dot_prod_dim"]
            model_settings["seft_latent_width"] = model_args["seft_latent_width"]
            model_settings["seft_n_rho_layers"] = model_args["seft_n_rho_layers"]
            model_settings["seft_rho_width"] = model_args["seft_rho_width"]
            model_settings["seft_rho_dropout"] = model_args["seft_rho_dropout"]
        if model_type in ("grud", "ipnets"):
            model_settings["recurrent_dropout"] = model_args["recurrent_dropout"]
            model_settings["recurrent_n_units"] = model_args["recurrent_n_units"]
        if model_type == "ipnets":
            model_settings["ipnets_imputation_stepsize"] = model_args["ipnets_imputation_stepsize"]
            model_settings["ipnets_reconst_fraction"] = model_args["ipnets_reconst_fraction"]
        if model_type == "EHRMamba2":
            # Instantiate Mamba2Pretrain with CLI-provided parameters
            # model_settings["vocab_size"] = model_args["vocab_size"]
            model_settings["embedding_size"] = model_args["embedding_size"]
            model_settings["max_num_visits"] = model_args["max_num_visits"]
            model_settings["max_seq_length"] = model_args["max_seq_length"]
            model_settings["state_size"] = model_args["state_size"]
            model_settings["num_heads"] = model_args["num_heads"]
            model_settings["head_dim"] = model_args["head_dim"]
            model_settings["num_hidden_layers"] = model_args["num_hidden_layers"]
            model_settings["expand"] = model_args["expand"]
            model_settings["conv_kernel"] = model_args["conv_kernel"]
            model_settings["dropout_prob"] = model_args["dropout_prob"]
            model_settings["padding_idx"] = model_args["padding_idx"]
            model_settings["cls_idx"] = model_args["cls_idx"]
            model_settings["eos_idx"] = model_args["eos_idx"]
            model_settings["n_groups"] = model_args["n_groups"]
            model_settings["chunk_size"] = model_args["chunk_size"]
        if model_type == "EHRMamba":
            model_settings["embedding_size"] = model_args["embedding_size"]
            model_settings["time_embeddings_size"] = model_args["time_embeddings_size"]
            model_settings["visit_order_size"] = 1
            model_settings["type_vocab_size"] = model_args["type_vocab_size"]
            model_settings["max_num_visits"] = model_args["max_num_visits"]
            model_settings["max_seq_length"] = model_args["max_seq_length"]
            model_settings["state_size"] = model_args["state_size"]
            model_settings["num_hidden_layers"] = model_args["num_hidden_layers"]
            model_settings["expand"] = model_args["expand"]
            model_settings["conv_kernel"] = model_args["conv_kernel"]
            model_settings["dropout_prob"] = model_args["dropout_prob"]
            model_settings["padding_idx"] = model_args["padding_idx"]
            model_settings["cls_idx"] = model_args["cls_idx"]



        with open(f"{run_path}/model_settings.json", "w") as fp:
            json.dump(model_settings, fp)

        # run training
        loss, accuracy_score, auprc_score, auc_score = train_test(
            train_pair,
            val_data,
            test_data,
            output_path=run_path,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            early_stop_criteria=early_stop_criteria,
            model_args=model_args,
        )

        accum_loss.append(loss)
        accum_accuracy.append(accuracy_score)
        accum_auprc.append(auprc_score)
        accum_auroc.append(auc_score)

    with open(f"{output_path}/summary.json", "w") as f:
        json.dump(
            {
                "mean_loss": float(np.mean(accum_loss)),
                "mean_accuracy": float(np.mean(accuracy_score)),
                "mean_auprc": float(np.mean(auprc_score)),
                "mean_auroc": float(np.mean(auc_score)),
                "std_loss": float(np.std(accum_loss)),
                "std_accuracy": float(np.std(accuracy_score)),
                "std_auprc": float(np.std(auprc_score)),
                "std_auroc": float(np.std(auc_score)),
            }, f, indent=4,
        )


if __name__ == "__main__":
    core_function()  
