import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from torch import nn
from sklearn import metrics
import json
import pandas as pd
from mortality_part_preprocessing import PairedDataset, MortalityDataset
from models.regular_transformer import EncoderClassifierRegular
from models.early_stopper import EarlyStopping
from models.deep_set_attention import DeepSetAttentionModel
from models.grud import GRUDModel
from models.ip_nets import InterpolationPredictionModel
from models.EHRMamba2 import Mamba2Pretrain
from models.EHRMamba import MambaPretrain
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def train_test(
    train_pair,
    val_data,
    test_data,
    output_path,
    model_type,
    model_args,
    batch_size=64,
    epochs=300,
    patience=5,
    lr=0.0001,
    early_stop_criteria="auroc"
):
    train_batch_size = batch_size // 2  # we concatenate 2 batches together

    train_collate_fn = PairedDataset.paired_collate_fn_truncate
    val_test_collate_fn = MortalityDataset.non_pair_collate_fn_truncate

    train_dataloader = DataLoader(train_pair, train_batch_size, shuffle=True, num_workers=0, collate_fn=train_collate_fn, pin_memory=False)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=False, num_workers=16, collate_fn=val_test_collate_fn, pin_memory=True)

    # assign GPU
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    val_loss, model = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_path=output_path,
        epochs=epochs,
        patience=patience,
        device=device,
        model_type=model_type,
        batch_size=batch_size,
        lr=lr,
        early_stop_criteria=early_stop_criteria,
        model_args=model_args
    )

    loss, accuracy_score, auprc_score, auc_score = test(
        test_dataloader=test_dataloader,
        output_path=output_path,
        device=device,
        model_type=model_type,
        model=model,
        model_args=model_args,
    )

    return loss, accuracy_score, auprc_score, auc_score

# # New added
def pad_features(features, max_dim):
    """
    Pads the feature dimension of the input tensor to the specified max_dim.

    Args:
        features (torch.Tensor): Input tensor of shape [batch_size, seq_length, feature_dim].
        max_dim (int): The desired maximum feature dimension.

    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, seq_length, max_dim].
    """
    # Calculate the amount of padding needed
    padding_dim = max_dim - features.shape[-1]
    if padding_dim > 0:
        # Create a padding tensor of zeros
        padding = torch.zeros(features.shape[0], features.shape[1], padding_dim, device=features.device)
        # Concatenate the padding to the features
        features = torch.cat((features, padding), dim=-1)
    elif padding_dim < 0:
        # Truncate the features to max_dim
        features = features[:, :, :max_dim]
    # If padding_dim == 0, no action is needed
    return features

def train(
    train_dataloader,
    val_dataloader,
    output_path,
    epochs,
    patience,
    device,
    model_type,
    lr,
    early_stop_criteria,
    model_args,
    **kwargs,  
):
    """
    training
    """

    iterable_inner_dataloader = iter(train_dataloader)
    test_batch = next(iterable_inner_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]

    # make a new model and train
    if model_type == "grud":
        model = GRUDModel(
            input_dim=sensor_count,
            static_dim=static_size,
            output_dims=2,
            device=device,
            **model_args
        )
    elif model_type == "ipnets":
        model = InterpolationPredictionModel(
            output_dims=2,
            sensor_count=sensor_count,
            **model_args
        )
    elif model_type == "seft":
        model = DeepSetAttentionModel(
            output_activation=None,
            n_modalities=sensor_count,
            output_dims=2,
            **model_args
        )
    elif model_type == "transformer":
        model = EncoderClassifierRegular(
            num_classes=2,
            device=device,
            max_timepoint_count=max_seq_length,
            sensors_count=sensor_count,
            static_count=static_size,
            return_intermediates=False,
            **model_args
        )
    elif model_type == "EHRMamba2":
        model = Mamba2Pretrain(
            vocab_size=model_args.get("vocab_size", 30522),  # Example default
            embedding_size=model_args.get("embedding_size", 768),
            max_num_visits=model_args.get("max_num_visits", 512),
            max_seq_length=model_args.get("max_seq_length", 2048),
            state_size=model_args.get("state_size", 64),
            num_heads=model_args.get("num_heads", 24),
            head_dim=model_args.get("head_dim", 64),
            num_hidden_layers=model_args.get("num_hidden_layers", 32),
            expand=model_args.get("expand", 2),
            conv_kernel=model_args.get("conv_kernel", 4),
            learning_rate=lr,
            dropout_prob=model_args.get("dropout_prob", 0.1),
            padding_idx=model_args.get("padding_idx", 0),
            cls_idx=model_args.get("cls_idx", 1),
            eos_idx=model_args.get("eos_idx", 2),
            n_groups=model_args.get("n_groups", 1),
            chunk_size=model_args.get("chunk_size", 256),
            # device=device
        )
    elif model_type == "EHRMamba":
        model = MambaPretrain(
            # num_heads=model_args.get("num_heads", 24),
            # head_dim=model_args.get("head_dim", 64),
            # eos_idx=model_args.get("eos_idx", 2),
            # n_groups=model_args.get("n_groups", 1),
            # chunk_size=model_args.get("chunk_size", 256)
            vocab_size=model_args.get("vocab_size", 30522),    # Example default
            embedding_size=model_args.get("embedding_size", 768),
            time_embeddings_size=model_args.get("time_embeddings_size", 32),
            visit_order_size=model_args.get("visit_order_size", 3),
            type_vocab_size=model_args.get("type_vocab_size", 9),
            max_num_visits=model_args.get("max_num_visits", 512),
            max_seq_length=model_args.get("max_seq_length", 2048),
            state_size=model_args.get("state_size", 16),
            num_hidden_layers=model_args.get("num_hidden_layers", 32),
            expand=model_args.get("expand", 2),
            conv_kernel=model_args.get("conv_kernel", 4),
            learning_rate=lr,
            dropout_prob=model_args.get("dropout_prob", 0.1),
            padding_idx=model_args.get("padding_idx", 0),
            cls_idx=model_args.get("cls_idx", 5),
            # device=device,
        )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"# of trainable parameters: {params}")
    if model_type == 'EHRMamba':
        criterion = nn.BCEWithLogitsLoss() # loss
    else:
        criterion = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )

    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=f"{output_path}/checkpoint.pt"
    )  # set up early stopping

    # initialize results file
    with open(f"{output_path}/training_log.csv", "w") as train_log:
        train_log.write(
            ",".join(["epoch", "train_loss", "val_loss", "val_roc_auc_score"]) + "\n"
        )

    for epoch in range(epochs):

        # training step

        model.train().to(device)  # sets training mode
        loss_list = []
        # for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
        #     data, times, static, labels, mask, delta = batch
        #     data = data.long()
            
        #     if model_type != "grud":
        #         data = data.to(device)
        #         static = static.to(device)
        #         times = times.to(device)
        #         mask = mask.to(device)
        #         delta = delta.to(device)

        #     optimizer.zero_grad()

        #     if model_type == "EHRMamba":
                
        #         inputs = (data, times, static, labels)  # Pack inputs as a tuple
        #         predictions = model(inputs)  # Pass the tuple to the model
        #     else:
        #         predictions = model(
        #             x=data, static=static, time=times, sensor_mask=mask, delta=delta
        #         )

        #     if type(predictions) == tuple:
        #         predictions, recon_loss = predictions
        #     else:
        #         recon_loss = 0
        #     predictions = predictions.squeeze(-1)
        #     loss = criterion(predictions.cpu(), labels) + recon_loss
        #     loss_list.append(loss.item())
        #     loss.backward()
        #     optimizer.step()
        for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
            data, times, static, labels, mask, delta = batch

            # Convert data to appropriate types and move to device
            if model_type != "grud":
                # data = data.long().to(device)
                data = data.to(device)
                static = static.to(device)
                times = times.to(device)
                mask = mask.to(device)
                delta = delta.to(device)

            optimizer.zero_grad()

            if model_type == "EHRMamba":
                labels = labels.to(device)
                # max_feature_dim = 200  # Ensure this matches your model's input_dim
                # data = pad_features(data, max_feature_dim)
                inputs = (data, times, static, labels)
                # outputs = model(inputs)
                predictions = model(inputs)  # Forward pass with labels
                # for i in range(len(predictions[0])):
                criterion = nn.CrossEntropyLoss()
                loss = criterion(predictions, labels)
                    # loss_list.append(loss.item())
                # loss = outputs["loss"]  # Access the loss from the dictionary
                # print('train loss:', loss)
 
            else:
                # For other model types, process as needed
                predictions = model(
                    x=data, static=static, time=times, sensor_mask=mask, delta=delta
                )
                if type(predictions) == tuple:
                    predictions, recon_loss = predictions
                else:
                    recon_loss = 0
                predictions = predictions.squeeze(-1)
                loss = criterion(predictions.cpu(), labels) + recon_loss
                # predictions = model(
                #     x=data, static=static, time=times, sensor_mask=mask, delta=delta
                # )
                # loss = criterion(predictions.squeeze(-1), labels)
                # if type(predictions) == tuple:
                #     predictions, recon_loss = predictions
                # else:
                #     recon_loss = 0
                # predictions = predictions.squeeze(-1)
                # loss = criterion(predictions.cpu(), labels) + recon_loss
            
            loss_list.append(loss.item())
            # print('output loss:', loss_list)
            loss.backward()
            optimizer.step()

            # # Backpropagation
            # loss.backward()
            # optimizer.step()

            # # Record loss
            # loss_list.append(loss.item())
        # print('train output loss:', loss_list)
        accum_loss = np.mean(loss_list)

        ######## validation step   #################################################################3
        model.eval().to(device)
        labels_list = torch.LongTensor([])
        predictions_list = torch.FloatTensor([])
        with torch.no_grad():
            for batch in val_dataloader:
                data, times, static, labels, mask, delta = batch
                labels_list = torch.cat((labels_list, labels), dim=0)
                if model_type != "grud":
                    data = data.to(device)
                    static = static.to(device)
                    times = times.to(device)
                    mask = mask.to(device)
                    delta = delta.to(device)

                if model_type == "EHRMamba":
                    labels = labels.to(device)
                    inputs = (data, times, static, labels)  # Pack inputs as a tuple
                    predictions = model(inputs)
                    # outputs = model(inputs)
                    # predictions = outputs["logits"]
                    # logits = outputs['logits']
                    # predictions = torch.argmax(logits, dim=1)
                    # predictions = outputs["preds"]
                    # print('output logits:', outputs["logits"])
                else:
                    predictions = model(
                        x=data, static=static, time=times, sensor_mask=mask, delta=delta
                    )
                # predictions = model(
                #     x=data, static=static, time=times, sensor_mask=mask, delta=delta
                # )
                    if type(predictions) == tuple:
                        predictions, _ = predictions
                    predictions = predictions.squeeze(-1)
                
                predictions_list = torch.cat(
                    (predictions_list, predictions.cpu()), dim=0
                )
                if model_type == "EHRMamba":
                    # probs = torch.sigmoid(predictions_list)
                    probs = torch.nn.functional.softmax(predictions_list, dim=1)
                    auc_score = metrics.roc_auc_score(labels_list, probs[:,1])
                    aupr_score = metrics.average_precision_score(labels_list, probs[:,1])

                else:
                    probs = torch.nn.functional.softmax(predictions_list, dim=1)
                    auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
                    aupr_score = metrics.average_precision_score(labels_list, probs[:, 1])
        # print('validation prediction list shape:', predictions_list.shape)
        # print('validation probs:', probs)
        if model_type == 'EHRMamba':
            criterion = nn.CrossEntropyLoss()
            val_loss = criterion(predictions_list.cpu(), labels_list)
        else:
            val_loss = criterion(predictions_list.cpu(), labels_list)

        with open(f"{output_path}/training_log.csv", "a") as train_log:
            train_log.write(
                ",".join(map(str, [epoch + 1, accum_loss, val_loss.item(), auc_score]))
                + "\n"
            )

        print(f"Epoch: {epoch+1}, Train Loss: {accum_loss}, Val Loss: {val_loss}")

        # set early stopping
        if early_stop_criteria == "auroc":
            early_stopping(1 - auc_score, model)
        elif early_stop_criteria == "auprc":
            early_stopping(1 - aupr_score, model)
        elif early_stop_criteria == "auprc+auroc":
            early_stopping(1 - (aupr_score + auc_score), model)
        elif early_stop_criteria == "loss":
            early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # save training curves
    training_log = pd.read_csv(f"{output_path}/training_log.csv")
    fig = plt.figure()
    fig.suptitle("training curves")
    ax0 = fig.add_subplot(121, title="loss")
    ax0.plot(training_log["train_loss"], label="Training")
    ax0.plot(training_log["val_loss"], label="Validation")
    ax0.legend()
    ax1 = fig.add_subplot(122, title="auroc")
    ax1.plot(training_log["val_roc_auc_score"], label="Training")
    ax1.legend()
    fig.savefig(f"{output_path}/train_curves.jpg")

    return val_loss, model


def test(
    test_dataloader,
    output_path,
    device,
    model_type,
    model,
    **kwargs,
):

    iterable_dataloader = iter(test_dataloader)
    test_batch = next(iterable_dataloader)
    max_seq_length = test_batch[0].shape[2]
    sensor_count = test_batch[0].shape[1]
    static_size = test_batch[2].shape[1]

    if model_type == 'EHRMamba':
        criterion = nn.BCEWithLogitsLoss() # loss
    else:
        criterion = nn.CrossEntropyLoss()  # loss
    # criterion = nn.CrossEntropyLoss()
    model.load_state_dict(
        torch.load(f"{output_path}/checkpoint.pt")
    )  # NEW: reload best model

    model.eval().to(device)

    labels_list = torch.LongTensor([])
    predictions_list = torch.FloatTensor([])
    with torch.no_grad():
        for batch in test_dataloader:
            data, times, static, labels, mask, delta = batch
            # data = data.long()
            labels_list = torch.cat((labels_list, labels), dim=0)
            if model_type != "grud":
                data = data.to(device)
                static = static.to(device)
                times = times.to(device)
                mask = mask.to(device)
                delta = delta.to(device)

            if model_type == "EHRMamba":
                labels = labels.to(device)
                # max_feature_dim = 200  # Ensure this matches your model's input_dim
                # data = pad_features(data, max_feature_dim)
                # inputs = (data, times, static, labels)  # Pack inputs as a tuple
                # predictions = model(inputs)  # Pass the tuple to the model
                inputs = (data, times, static, labels)  # Pack inputs as a tuple
                predictions = model(inputs)
                # outputs = model(inputs, labels=labels, return_dict=True)
                # logits = outputs["logits"]
                # predictions = torch.argmax(logits, dim=1)
                # predictions = torch.sigmoid(outputs["logits"])  # Example for binary classification
                # predictions = outputs["logits"]
                # print('output logits:', outputs["logits"])
                # print('output preds:', outputs["predictions"])

            else:
                predictions = model(
                    x=data, static=static, time=times, sensor_mask=mask, delta=delta
                )
                # print('output predictions:', predictions)
            # predictions = model(
            #     x=data, static=static, time=times, sensor_mask=mask, delta=delta
            # )
                if type(predictions) == tuple:
                    predictions, _ = predictions
                predictions = predictions.squeeze(-1)
            predictions_list = torch.cat((predictions_list, predictions.cpu()), dim=0)
    # print('test output prediction list:', predictions_list)
    if model_type == 'EHRMamba':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions_list.cpu(), labels_list)
    else:
        loss = criterion(predictions_list.cpu(), labels_list)
    # loss = criterion(predictions_list.cpu(), labels_list)
    print(f"Test Loss: {loss}")

    if model_type == "EHRMamba":
        # probs = torch.sigmoid(predictions_list)
        probs = torch.nn.functional.softmax(predictions_list, dim=1)
        # preds = (probs > 0.5).int()  # Binary predictions (0 or 1)
        results = metrics.classification_report(
        labels_list.cpu(), torch.argmax(probs.cpu(), dim=1).cpu(), output_dict=True  # predictions_list
    )
        cm = metrics.confusion_matrix(
        labels_list.cpu(), torch.argmax(probs.cpu(), dim=1).cpu()
    )
        # results = metrics.classification_report(labels_list.cpu(), preds.cpu(), output_dict=True)
        # cm = metrics.confusion_matrix(labels_list.cpu(), preds.cpu())
    else:
        probs = torch.nn.functional.softmax(predictions_list, dim=1)
        results = metrics.classification_report(
        labels_list, torch.argmax(probs, dim=1), output_dict=True  # predictions_list
    )
        cm = metrics.confusion_matrix(
        labels_list, torch.argmax(probs, dim=1)
    )

    # results = metrics.classification_report(
    #     labels_list, torch.argmax(probs, dim=1), output_dict=True  # predictions_list
    # )
    # cm = metrics.confusion_matrix(
    #     labels_list, torch.argmax(probs, dim=1)
    # )

    if model_type == "EHRMamba":
        # auc_score = metrics.roc_auc_score(labels_list.cpu(), probs.cpu())
        # auprc_score = metrics.average_precision_score(labels_list.cpu(), probs.cpu())
        # accuracy_score = metrics.accuracy_score(labels_list.cpu(), preds.cpu())
        auc_score = metrics.roc_auc_score(labels_list.cpu(), probs[:, 1].cpu())
        auprc_score = metrics.average_precision_score(labels_list.cpu(), probs[:, 1].cpu())
        accuracy_score = metrics.accuracy_score(labels_list.cpu(), np.argmax(probs.cpu(), axis=1).cpu())
    else:
        auc_score = metrics.roc_auc_score(labels_list, probs[:, 1])
        auprc_score = metrics.average_precision_score(labels_list, probs[:, 1])
        accuracy_score = metrics.accuracy_score(labels_list, np.argmax(probs, axis=1))

    print(results)
    print(cm)
    print(f"Accuracy = {accuracy_score}")
    print(f"AUPRC = {auprc_score}")
    print(f"AUROC = {auc_score}")

    # save test metrics
    test_metrics = {
        "test_loss": loss.item(),
        "accuracy": accuracy_score,
        "AUPRC": auprc_score,
        "AUROC": auc_score,
    }
    test_metrics.update(results)
    # test_metrics.update(cm) # TO DO: add later
    with open(f"{output_path}/test_results.json", "w") as fp:
        json.dump(test_metrics, fp)

    return loss, accuracy_score, auprc_score, auc_score
