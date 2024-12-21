"""Mamba model."""

# from typing import Any, Dict, Optional, Tuple, Union
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import (
    MambaCausalLMOutput,
    MambaForCausalLM,
)

from models.mamba_utils import (
    MambaForMultiHeadSequenceClassification,
    MambaForSequenceClassification,
    MambaSequenceClassifierOutput,
)
from models.embeddings import MambaEmbeddingsForCEHR


class MambaPretrain(pl.LightningModule):
    """Mamba model for pretraining."""

    def __init__(
        self,
        # device,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
        input_dim: int = 200,
    ):
        super().__init__()

        # assign GPU
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        self.device_type = torch.device(dev)  # Rename `device` to `device_type`
        self.test_outputs = []


        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.max_seq_length = max_seq_length
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.input_dim = input_dim


        self.config = MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.padding_idx,
        )
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            input_dim=self.input_dim,
            type_vocab_size=self.type_vocab_size,
            max_num_visits=self.max_num_visits,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
            hidden_dropout_prob=self.dropout_prob,
        ).to(self.device_type)
        # Initialize weights and apply final processing
        self.post_init()

        # Mamba has its own initialization
        self.model = MambaForCausalLM(config=self.config).to(self.device_type)

        # self.classification_head = nn.Linear(self.embedding_size, 1)  # Binary classification
        
        self.sigmoid = nn.Sigmoid().to(self.device_type)   # For output probabilities (optional, depends on loss function)


    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor]
        ],
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""
        # concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        # inputs_embeds = self.embeddings(
        #     input_ids=concept_ids,
        #     token_type_ids_batch=type_ids,
        #     time_stamps=time_stamps,
        #     ages=ages,
        #     visit_orders=visit_orders,
        #     visit_segments=visit_segments,
        # )

        # For P12 dataset
            # Unpack inputs
        if len(inputs) == 4:  # Expecting labels
            data, times, static, labels = inputs
        elif len(inputs) == 3:  # Without labels
            data, times, static = inputs
        else:
            raise ValueError(f"Unexpected number of inputs: {len(inputs)}")
        # data, times, static, labels= inputs
        # data, times, static= inputs

        # Change data into shape[batches, numOfFeatures*featureValue]
        input_ids = data.to(self.device_type)
        # input_ids_transposed = input_ids.permute(0, 2, 1)
        # input_ids_reshaped = input_ids_transposed.reshape(64, -1)
        # input_ids = input_ids_reshaped
        time_stamps = times.to(self.device_type)
        ages = static[:, 0].to(self.device_type) # Only use the first column of static for ages
        if labels is not None:
            labels = labels.to(self.device_type)

        # labels_tensor = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        visit_segments = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        visit_orders = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)

        inputs_embeds = self.embeddings(
            input_ids=input_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_segments=visit_segments,
            visit_orders=visit_orders,
        )

        # print('embeds size:', inputs_embeds.shape)
        # print('labels size:', labels.shape)

        # loss = outputs[0]
        # logits = outputs[1]
        # print('mamba outputs:', outputs)
        hidden_states = self.model(inputs_embeds=inputs_embeds, return_dict=True).hidden_states
        outputs = self.model(inputs_embeds=inputs_embeds, return_dict=True)
        # print('output size in Mamba:', outputs['logits'].shape)
        logits = outputs['logits']
        pooled_logits = logits.mean(dim=1)
        classification_head = nn.Linear(logits.size(-1), 2).to(self.device_type)
        predictions = classification_head(pooled_logits)
        # print("Final logits for binary classification:", logits)
        # print("Final logits shape for binary classification:", logits.shape)
        # predictions = torch.argmax(logits, dim=1)
        # Extract the last hidden state (if needed)
        # hidden_states = outputs.hidden_states[-1] if output_hidden_states else outputs[0]
        # # print(f"hidden_states shape: {hidden_states.shape}")

        # # Apply classification head
        # embeddingSize = hidden_states.size(-1) 
        # classification_head = nn.Linear(embeddingSize, 1).to(self.device_type)
        # logits = classification_head(hidden_states.mean(dim=1))  # Mean-pool across sequence
        # print('logits:', logits)
        # Pooling to align with binary classification
        # pooled_hidden_states = hidden_states.mean(dim=1)  # Mean-pool across the sequence
        # logits = self.classification_head(pooled_hidden_states)  # Apply classification head
        # logits = logits.squeeze(-1)  # Remove the last dimension
        # print(f"logits shape: {logits.shape}, labels shape: {labels.shape}")

        # loss = None
        # if labels is not None:
        #     # Calculate loss for classification
        #     loss_fct = nn.BCEWithLogitsLoss()
        #     # loss = loss_fct(logits.view(-1), labels.view(-1).float())
        #     loss = loss_fct(logits.squeeze(-1), labels.float())  # Access logits from the dictionary
        #     # return loss, logits
        
        return predictions
        
        # if return_dict:
        #     return {"loss": loss, "logits": logits}
        # else:
        #     return (loss, logits) if loss is not None else logits

        # if labels is not None:
        # # Align labels with inputs
        #     if labels.size(0) != inputs_embeds.size(0):
        #         print(f"Aligning Labels: {labels.size(0)} vs {inputs_embeds.size(0)}")
        #         labels = labels[:inputs_embeds.size(0)]

        # return self.model(
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # return loss, logits
        # return {"loss": loss, "logits": logits}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
        # inputs = (
        #     batch["concept_ids"],
        #     batch["type_ids"],
        #     batch["time_stamps"],
        #     batch["ages"],
        #     batch["visit_orders"],
        #     batch["visit_segments"],
        # )
        # For dataset P12
        # data, times, static, labels, _, _ = batch
        # inputs = (data, times, static, labels)

        # logits = self(inputs)
        # loss_fct = nn.BCEWithLogitsLoss()
        # loss = loss_fct(logits.view(-1), labels.float())

        # self.log("train_loss", loss)

        inputs = (
            batch["data"],
            batch["times"],
            batch["static"],
        )
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                return_dict=True,
            ).loss

        # (current_lr,) = self.lr_schedulers().get_last_lr()
        # self.log_dict(
        #     dictionary={"train_loss": loss, "lr": current_lr},
        #     on_step=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
        # inputs = (
        #     batch["concept_ids"],
        #     batch["type_ids"],
        #     batch["time_stamps"],
        #     batch["ages"],
        #     batch["visit_orders"],
        #     batch["visit_segments"],
        # )
        # For dataset P12
        # data, times, static, labels, mask, delta = batch
        # inputs = (data, times, static)  

        # logits = self(inputs)
        # loss_fct = nn.BCEWithLogitsLoss()
        # loss = loss_fct(logits.view(-1), labels.float())
        # self.log("val_loss", loss)

        inputs = (
            batch["data"],
            batch["times"],
            batch["static"],
        )
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            outputs = self(
                inputs,
                labels=labels,
                return_dict=True,
            )

        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=1)
        log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}

        # (current_lr,) = self.lr_schedulers().get_last_lr()
        # self.log_dict(
        #     dictionary={"val_loss": loss, "lr": current_lr},
        #     on_step=True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )
        return log

    def configure_optimizers(
        self,
    # ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
    ) -> Tuple[List[Any], List[Dict[str, Union[SequentialLR, str]]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=n_decay_steps,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Test step."""
        # inputs = (
        #     batch["concept_ids"],
        #     batch["type_ids"],
        #     batch["time_stamps"],
        #     batch["ages"],
        #     batch["visit_orders"],
        #     batch["visit_segments"],
        # )
        # For dataset P12
        inputs = (
            batch["data"],
            batch["times"],
            batch["static"],
        )
        labels = batch["labels"]
        # task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            outputs = self(
                inputs,
                labels=labels,
                # task_indices=task_indices,
                return_dict=True,
            )
            print('outputs in test:', outputs)
            logits = outputs
            logits = self(inputs)  # Get logits directly for classification

        preds = (torch.sigmoid(logits) > 0.5).int()  # Convert logits to binary predictions
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1), labels.float())  # Binary cross-entropy loss

        log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}
        self.test_outputs.append(log)  # Append to instance attribute for epoch-end logging

        # loss = outputs[0]
        # logits = outputs[1]
        # preds = torch.argmax(logits, dim=1)
        # log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}

        # Append the outputs to the instance attribute
        # self.test_outputs.append(log)

        return log

    def on_test_epoch_end(self) -> Any:
        """Evaluate after the test epoch."""
        labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
        preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
        loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
        logits = torch.cat([x["logits"] for x in self.test_outputs]).cpu()

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, torch.sigmoid(logits))  # Use sigmoid for probability
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        # auprc = average_precision_score(labels, torch.sigmoid(logits))  # Probabilities for AUPRC


        # # Update the saved outputs to include all concatanted batches
        # self.test_outputs = {
        #     "labels": labels,
        #     "logits": logits,
        # }

        # if self.config.problem_type == "multi_label_classification":
        #     preds_one_hot = np.eye(labels.shape[1])[preds]
        #     accuracy = accuracy_score(labels, preds_one_hot)
        #     f1 = f1_score(labels, preds_one_hot, average="micro")
        #     auc = roc_auc_score(labels, preds_one_hot, average="micro")
        #     precision = precision_score(labels, preds_one_hot, average="micro")
        #     recall = recall_score(labels, preds_one_hot, average="micro")

        # else:  # single_label_classification
        #     accuracy = accuracy_score(labels, preds)
        #     f1 = f1_score(labels, preds)
        #     auc = roc_auc_score(labels, preds)
        #     precision = precision_score(labels, preds)
        #     recall = recall_score(labels, preds)

        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        # self.log("test_auprc", auprc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

        # Update instance attribute for final outputs
        self.test_outputs = {
            "labels": labels,
            "preds": preds,
            "logits": logits,
        }

        return loss


# class MambaFinetune(pl.LightningModule):
#     """Mamba model for fine-tuning."""

#     def __init__(
#         self,
#         pretrained_model: MambaPretrain,
#         problem_type: str = "single_label_classification",
#         num_labels: int = 2,
#         num_tasks: int = 6,
#         learning_rate: float = 5e-5,
#         classifier_dropout: float = 0.1,
#         multi_head: bool = False,
#     ):
#         super().__init__()

#         self.num_labels = num_labels
#         self.num_tasks = num_tasks
#         self.learning_rate = learning_rate
#         self.classifier_dropout = classifier_dropout
#         self.multi_head = multi_head
#         self.test_outputs = []

#         self.config = pretrained_model.config
#         self.config.num_labels = self.num_labels
#         self.config.classifier_dropout = self.classifier_dropout
#         self.config.problem_type = problem_type

#         if self.multi_head:
#             self.model = MambaForMultiHeadSequenceClassification(
#                 config=self.config, num_tasks=self.num_tasks
#             )
#         else:
#             self.model = MambaForSequenceClassification(config=self.config)

#         # self.post_init()

#         self.pretrained_model = pretrained_model
#         self.embeddings = self.pretrained_model.embeddings
#         self.model.backbone = self.pretrained_model.model.backbone

#     def _init_weights(self, module: torch.nn.Module) -> None:
#         """Initialize the weights."""
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def post_init(self) -> None:
#         """Apply weight initialization."""
#         self.apply(self._init_weights)

#     def forward(
#         self,
#         inputs: Tuple[
#             torch.Tensor,
#             torch.Tensor,
#             torch.Tensor,
#             torch.Tensor,
#             # torch.Tensor,
#             # torch.Tensor,
#         ],
#         labels: Optional[torch.Tensor] = None,
#         task_indices: Optional[torch.Tensor] = None,
#         output_hidden_states: Optional[bool] = False,
#         return_dict: Optional[bool] = True,
#     ) -> Union[Tuple[torch.Tensor, ...], MambaSequenceClassifierOutput]:
#         """Forward pass for the model."""
#         # concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
#         # inputs_embeds = self.embeddings(
#         #     input_ids=concept_ids,
#         #     token_type_ids_batch=type_ids,
#         #     time_stamps=time_stamps,
#         #     ages=ages,
#         #     visit_orders=visit_orders,
#         #     visit_segments=visit_segments,
#         # )
#         # For dataset P12
#         data, times, static, labels= inputs
#         inputs_embeds = self.embeddings(
#             input_ids=data,
#             time_stamps=times,
#             ages=static[:, 0],  # Only use the first column of static for ages
#         )

#         return self.model(
#             # input_ids=concept_ids,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             task_indices=task_indices,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#     def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
#         """Train model on training dataset."""
#         # inputs = (
#         #     batch["concept_ids"],
#         #     batch["type_ids"],
#         #     batch["time_stamps"],
#         #     batch["ages"],
#         #     batch["visit_orders"],
#         #     batch["visit_segments"],
#         # )
#         # For dataset P12
#         inputs = (
#             batch["data"],
#             batch["times"],
#             batch["static"],
#         )
#         labels = batch["labels"]
#         task_indices = batch["task_indices"]

#         # Ensure use of mixed precision
#         with autocast():
#             loss = self(
#                 inputs,
#                 labels=labels,
#                 task_indices=task_indices,
#                 return_dict=True,
#             ).loss

#         (current_lr,) = self.lr_schedulers().get_last_lr()
#         self.log_dict(
#             dictionary={"train_loss": loss, "lr": current_lr},
#             on_step=True,
#             prog_bar=True,
#             sync_dist=True,
#         )

#         return loss

#     def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
#         """Evaluate model on validation dataset."""
#         # inputs = (
#         #     batch["concept_ids"],
#         #     batch["type_ids"],
#         #     batch["time_stamps"],
#         #     batch["ages"],
#         #     batch["visit_orders"],
#         #     batch["visit_segments"],
#         # )
#         # For dataset P12
#         inputs = (
#             batch["data"],
#             batch["times"],
#             batch["static"],
#         )
#         labels = batch["labels"]
#         task_indices = batch["task_indices"]

#         # Ensure use of mixed precision
#         with autocast():
#             loss = self(
#                 inputs,
#                 labels=labels,
#                 task_indices=task_indices,
#                 return_dict=True,
#             ).loss

#         (current_lr,) = self.lr_schedulers().get_last_lr()
#         self.log_dict(
#             dictionary={"val_loss": loss, "lr": current_lr},
#             on_step=True,
#             prog_bar=True,
#             sync_dist=True,
#         )

#         return loss

#     def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
#         """Test step."""
#         # inputs = (
#         #     batch["concept_ids"],
#         #     batch["type_ids"],
#         #     batch["time_stamps"],
#         #     batch["ages"],
#         #     batch["visit_orders"],
#         #     batch["visit_segments"],
#         # )
#         # For dataset P12
#         inputs = (
#             batch["data"],
#             batch["times"],
#             batch["static"],
#         )
#         labels = batch["labels"]
#         task_indices = batch["task_indices"]

#         # Ensure use of mixed precision
#         with autocast():
#             outputs = self(
#                 inputs,
#                 labels=labels,
#                 task_indices=task_indices,
#                 return_dict=True,
#             )

#         loss = outputs[0]
#         logits = outputs[1]
#         preds = torch.argmax(logits, dim=1)
#         log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}

#         # Append the outputs to the instance attribute
#         self.test_outputs.append(log)

#         return log

#     def on_test_epoch_end(self) -> Any:
#         """Evaluate after the test epoch."""
#         labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
#         preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
#         loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
#         logits = torch.cat([x["logits"] for x in self.test_outputs]).cpu()

#         # Update the saved outputs to include all concatanted batches
#         self.test_outputs = {
#             "labels": labels,
#             "logits": logits,
#         }

#         if self.config.problem_type == "multi_label_classification":
#             preds_one_hot = np.eye(labels.shape[1])[preds]
#             accuracy = accuracy_score(labels, preds_one_hot)
#             f1 = f1_score(labels, preds_one_hot, average="micro")
#             auc = roc_auc_score(labels, preds_one_hot, average="micro")
#             precision = precision_score(labels, preds_one_hot, average="micro")
#             recall = recall_score(labels, preds_one_hot, average="micro")

#         else:  # single_label_classification
#             accuracy = accuracy_score(labels, preds)
#             f1 = f1_score(labels, preds)
#             auc = roc_auc_score(labels, preds)
#             precision = precision_score(labels, preds)
#             recall = recall_score(labels, preds)

#         self.log("test_loss", loss)
#         self.log("test_acc", accuracy)
#         self.log("test_f1", f1)
#         self.log("test_auc", auc)
#         self.log("test_precision", precision)
#         self.log("test_recall", recall)

#         return loss

#     def configure_optimizers(
#         self,
#     # ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
#     ) -> Tuple[list[Any], list[dict[str, Union[SequentialLR, str]]]]:
#         """Configure optimizers and learning rate scheduler."""
#         optimizer = AdamW(
#             self.parameters(),
#             lr=self.learning_rate,
#         )

#         n_steps = self.trainer.estimated_stepping_batches
#         n_warmup_steps = int(0.1 * n_steps)
#         n_decay_steps = int(0.9 * n_steps)

#         warmup = LinearLR(
#             optimizer,
#             start_factor=0.01,
#             end_factor=1.0,
#             total_iters=n_warmup_steps,
#         )
#         decay = LinearLR(
#             optimizer,
#             start_factor=1.0,
#             end_factor=0.01,
#             total_iters=n_decay_steps,
#         )
#         scheduler = SequentialLR(
#             optimizer=optimizer,
#             schedulers=[warmup, decay],
#             milestones=[n_warmup_steps],
#         )

#         return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
