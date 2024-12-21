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
        use_mambapy: bool = False,
    ):
        super().__init__()

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
        self.use_mambapy = use_mambapy

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
            use_mambapy=self.use_mambapy,
        )
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            type_vocab_size=self.type_vocab_size,
            max_num_visits=self.max_num_visits,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
            hidden_dropout_prob=self.dropout_prob,
        )
        # Initialize weights and apply final processing
        self.post_init()

        # Mamba has its own initialization
        self.model = MambaForCausalLM(config=self.config)

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
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        if labels is None:
            labels = concept_ids

        return self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
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


class MambaFinetune(pl.LightningModule):
    """Mamba model for fine-tuning."""

    def __init__(
        self,
        pretrained_model: MambaPretrain,
        problem_type: str = "single_label_classification",
        num_labels: int = 2,
        num_tasks: int = 6,
        learning_rate: float = 5e-5,
        classifier_dropout: float = 0.1,
        multi_head: bool = False,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.num_tasks = num_tasks
        self.learning_rate = learning_rate
        self.classifier_dropout = classifier_dropout
        self.multi_head = multi_head
        self.test_outputs = []

        self.config = pretrained_model.config
        self.config.num_labels = self.num_labels
        self.config.classifier_dropout = self.classifier_dropout
        self.config.problem_type = problem_type

        if self.multi_head:
            self.model = MambaForMultiHeadSequenceClassification(
                config=self.config, num_tasks=self.num_tasks
            )
        else:
            self.model = MambaForSequenceClassification(config=self.config)

        # self.post_init()

        self.pretrained_model = pretrained_model
        self.embeddings = self.pretrained_model.embeddings
        self.model.backbone = self.pretrained_model.model.backbone

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
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        labels: Optional[torch.Tensor] = None,
        task_indices: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaSequenceClassifierOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        return self.model(
            input_ids=concept_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            task_indices=task_indices,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                task_indices=task_indices,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                task_indices=task_indices,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Test step."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            outputs = self(
                inputs,
                labels=labels,
                task_indices=task_indices,
                return_dict=True,
            )

        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=1)
        log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}

        # Append the outputs to the instance attribute
        self.test_outputs.append(log)

        return log

    def on_test_epoch_end(self) -> Any:
        """Evaluate after the test epoch."""
        labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
        preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
        loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
        logits = torch.cat([x["logits"] for x in self.test_outputs]).cpu()

        # Update the saved outputs to include all concatanted batches
        self.test_outputs = {
            "labels": labels,
            "logits": logits,
        }

        if self.config.problem_type == "multi_label_classification":
            preds_one_hot = np.eye(labels.shape[1])[preds]
            accuracy = accuracy_score(labels, preds_one_hot)
            f1 = f1_score(labels, preds_one_hot, average="micro")
            auc = roc_auc_score(labels, preds_one_hot, average="micro")
            precision = precision_score(labels, preds_one_hot, average="micro")
            recall = recall_score(labels, preds_one_hot, average="micro")

        else:  # single_label_classification
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            auc = roc_auc_score(labels, preds)
            precision = precision_score(labels, preds)
            recall = recall_score(labels, preds)

        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
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
    



# Old version of embeddings.py
"""Embedding layers for the models."""

import math
from typing import Any, Optional

import torch
from torch import nn
from transformers import BigBirdConfig, MambaConfig


class TimeEmbeddingLayer(nn.Module):
    """Embedding layer for time features."""

    def __init__(self, embedding_size: int, is_time_delta: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta

        self.w = nn.Parameter(torch.empty(1, self.embedding_size))
        self.phi = nn.Parameter(torch.empty(1, self.embedding_size))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.phi)

    def forward(self, time_stamps: torch.Tensor) -> Any:
        """Apply time embedding to the input time stamps."""
        if self.is_time_delta:
            # If the time_stamps represent time deltas, we calculate the deltas.
            # This is equivalent to the difference between consecutive elements.
            time_stamps = torch.cat(
                (time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]),
                dim=-1,
            )
        time_stamps = time_stamps.float()
        time_stamps_expanded = time_stamps.unsqueeze(-1)
        next_input = time_stamps_expanded * self.w + self.phi

        return torch.sin(next_input)


class VisitEmbedding(nn.Module):
    """Embedding layer for visit segments."""

    def __init__(
        self,
        visit_order_size: int,
        embedding_size: int,
    ):
        super().__init__()
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.visit_order_size, self.embedding_size)

    def forward(self, visit_segments: torch.Tensor) -> Any:
        """Apply visit embedding to the input visit segments."""
        return self.embedding(visit_segments)


class ConceptEmbedding(nn.Module):
    """Embedding layer for event concepts."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        padding_idx: Optional[int] = None,
    ):
        super(ConceptEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_size,
            padding_idx=padding_idx,
        )

    def forward(self, inputs: torch.Tensor) -> Any:
        """Apply concept embedding to the input concepts."""
        return self.embedding(inputs)


class PositionalEmbedding(nn.Module):
    """Positional embedding layer."""

    def __init__(self, embedding_size: int, max_len: int):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, embedding_size, 2).float()
            * -(math.log(10000.0) / embedding_size)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, visit_orders: torch.Tensor) -> Any:
        """Apply positional embedding to the input visit orders."""
        first_visit_concept_orders = visit_orders[:, 0:1]
        normalized_visit_orders = torch.clamp(
            visit_orders - first_visit_concept_orders,
            0,
            self.pe.size(0) - 1,
        )
        return self.pe[normalized_visit_orders]


class BERTEmbeddingsForCEHR(nn.Module):
    """Embeddings for CEHR-BERT."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 128,
        time_embeddings_size: int = 16,
        type_vocab_size: int = 9,
        visit_order_size: int = 3,
        max_len: int = 512,
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 0.1,
        padding_idx: int = 1,
    ):
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.concept_embedding = ConceptEmbedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            padding_idx=padding_idx,
        )
        self.token_type_embeddings = nn.Embedding(
            type_vocab_size,
            embedding_size,
        )
        self.time_embedding = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embedding = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
        )
        self.positional_embedding = PositionalEmbedding(
            embedding_size=embedding_size,
            max_len=max_len,
        )
        self.visit_embedding = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=embedding_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            embedding_size + 2 * time_embeddings_size,
            embedding_size,
        )  # Assuming 4 input features are concatenated
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        concept_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> Any:
        """Apply embeddings to the input features."""
        concept_embed = self.concept_embedding(concept_ids)
        type_embed = self.token_type_embeddings(type_ids)
        time_embed = self.time_embedding(time_stamps)
        age_embed = self.age_embedding(ages)
        positional_embed = self.positional_embedding(visit_orders)
        visit_segment_embed = self.visit_embedding(visit_segments)

        order_sequence_all = torch.arange(
            self.max_len, device=concept_ids.device
        ).expand_as(concept_ids)
        padding_mask = concept_ids == self.padding_idx
        order_sequence = torch.where(
            padding_mask,
            torch.tensor(self.max_len, device=concept_ids.device),
            order_sequence_all,
        )
        global_position_embed = self.positional_embedding(order_sequence)

        embeddings = torch.cat((concept_embed, time_embed, age_embed), dim=-1)
        embeddings = self.tanh(self.scale_back_concat_layer(embeddings))
        embeddings = (
            embeddings
            + type_embed
            + positional_embed
            + visit_segment_embed
            + global_position_embed
        )
        embeddings = self.LayerNorm(embeddings)

        return self.dropout(embeddings)


class BigBirdEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: BigBirdConfig,
        time_embeddings_size: int = 16,
        visit_order_size: int = 3,
    ) -> None:
        """Initiate wrapper class for embeddings used in BigBird CEHR classes."""
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        self.visit_order_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
        )
        self.visit_segment_embeddings = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=config.hidden_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            config.hidden_size + 2 * time_embeddings_size,
            config.hidden_size,
        )

        self.time_stamps: Optional[torch.Tensor] = None
        self.ages: Optional[torch.Tensor] = None
        self.visit_orders: Optional[torch.Tensor] = None
        self.visit_segments: Optional[torch.Tensor] = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory.
        self.position_embedding_type = getattr(
            config,
            "position_embedding_type",
            "absolute",
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )
        # End copy

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def cache_input(
        self,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
    ) -> None:
        """Cache values for time_stamps, ages, visit_orders & visit_segments.

        These values will be used by the forward pass to change the final embedding.

        Parameters
        ----------
        time_stamps : torch.Tensor
            Time stamps of the input data.
        ages : torch.Tensor
            Ages of the input data.
        visit_orders : torch.Tensor
            Visit orders of the input data.
        visit_segments : torch.Tensor
            Visit segments of the input data.
        """
        self.time_stamps = time_stamps
        self.ages = ages
        self.visit_orders = visit_orders
        self.visit_segments = visit_segments

    def clear_cache(self) -> None:
        """Delete the tensors cached by cache_input method."""
        del self.time_stamps, self.ages, self.visit_orders, self.visit_segments

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> Any:
        """Return the final embeddings of concept ids using input and cached values."""
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :,
                past_key_values_length : seq_length + past_key_values_length,
            ]

        # Setting the token_type_ids to the registered buffer in constructor
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0],
                    seq_length,
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

        # Using cached values from a prior cache_input call
        time_stamps_embeds = self.time_embeddings(self.time_stamps)
        ages_embeds = self.age_embeddings(self.ages)
        visit_segments_embeds = self.visit_segment_embeddings(self.visit_segments)
        visit_order_embeds = self.visit_order_embeddings(self.visit_orders)

        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        inputs_embeds = torch.cat(
            (inputs_embeds, time_stamps_embeds, ages_embeds),
            dim=-1,
        )
        inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        embeddings = inputs_embeds + token_type_embeds
        embeddings += position_embeds
        embeddings += visit_order_embeds
        embeddings += visit_segments_embeds

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)

        # Clear the cache for next forward call
        self.clear_cache()

        return embeddings


class MambaEmbeddingsForCEHR(nn.Module):
    """Construct the embeddings from concept, token_type, etc., embeddings."""

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(
        self,
        config: MambaConfig,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
        embedding_size=768,
    ) -> None:
        """Initiate wrapper class for embeddings used in Mamba CEHR classes."""
        super().__init__()
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.visit_order_size = visit_order_size

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(
            self.type_vocab_size,
            config.hidden_size,
        )
        self.visit_order_embeddings = nn.Embedding(
            self.max_num_visits,
            config.hidden_size,
        )
        self.time_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            is_time_delta=True,
        )
        self.age_embeddings = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
        )
        self.visit_segment_embeddings = VisitEmbedding(
            visit_order_size=visit_order_size,
            embedding_size=config.hidden_size,
        )
        self.scale_back_concat_layer = nn.Linear(
            config.hidden_size + 2 * time_embeddings_size,
            config.hidden_size,
        )
        # self.scale_back_concat_layer = nn.Linear(190, embedding_size)  # Use 156 as the input dimension


        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file.
        self.tanh = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        # End copy

    # def forward(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     inputs_embeds: Optional[torch.Tensor] = None,
    #     token_type_ids_batch: Optional[torch.Tensor] = None,
    #     time_stamps: Optional[torch.Tensor] = None,
    #     ages: Optional[torch.Tensor] = None,
    #     visit_orders: Optional[torch.Tensor] = None,
    #     visit_segments: Optional[torch.Tensor] = None,
    # ) -> Any:
    def forward(
        self,
        # data: torch.Tensor,
        # time_stamps: torch.Tensor,
        # ages: torch.Tensor,
        # token_type_ids_batch: Optional[torch.Tensor] = None,
        # visit_orders: Optional[torch.Tensor] = None,
        # visit_segments: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        token_type_ids_batch: Optional[torch.Tensor] = None,
        visit_orders: Optional[torch.Tensor] = None,
        visit_segments: Optional[torch.Tensor] = None,
    # ) -> Any:
    ) -> torch.Tensor:
        """Return the final embeddings of concept ids.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input data (concept_ids) to be embedded.
        inputs_embeds : torch.Tensor
            The embeddings of the input data.
        token_type_ids_batch : torch.Tensor
            The token type IDs of the input data.
        time_stamps : torch.Tensor
            Time stamps of the input data.
        ages : torch.Tensor
            Ages of the input data.
        visit_orders : torch.Tensor
            Visit orders of the input data.
        visit_segments : torch.Tensor
            Visit segments of the input data.
        """
        """Return the final embeddings of concept ids using P12 dataset variables."""
        # Map P12 variables to the original variables
        # if input_ids is None:
        #     input_ids = data  # Time-series data (vitals)
        #     input_ids = input_ids.long()
        # if inputs_embeds is None:
        #     inputs_embeds = self.word_embeddings(input_ids)  # Embedding of time-series data
        
        # input_ids = data.long()
        print("input_ids.shape:", input_ids.shape)
        print("input_ids max value:", input_ids.max())
        print("input_ids min value:", input_ids.min())
        print("Embedding vocab_size:", self.word_embeddings.num_embeddings)

        inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids_batch is None:
            token_type_ids_batch = torch.zeros_like(input_ids).long()

        if visit_orders is None:
            visit_orders = torch.zeros_like(input_ids).long()

        if visit_segments is None:
            visit_segments = torch.zeros_like(input_ids).long()


        # if token_type_ids_batch is None:
        #     token_type_ids_batch = torch.zeros_like(data[:, :, 0]).long()  # No token type distinction in P12

        if time_stamps is None:
            time_stamps = time  # Time since ICU admission

        if ages is None:
            ages = static[:, 0]  # Age extracted from static features

        # if visit_orders is None:
        #     visit_orders = torch.zeros(data.shape[0], dtype=torch.long)  # Single visit per patient

        # if visit_segments is None:
        #     visit_segments = torch.zeros(data.shape[0], dtype=torch.long)  # No segmentation within visits


        # Using cached values from a prior cache_input call
        time_stamps_embeds = self.time_embeddings(time_stamps)
        ages_embeds = self.age_embeddings(ages)
        visit_segments_embeds = self.visit_segment_embeddings(visit_segments)
        visit_order_embeds = self.visit_order_embeddings(visit_orders)
        token_type_embeds = self.token_type_embeddings(token_type_ids_batch)

        inputs_embeds = torch.cat(
            (inputs_embeds, time_stamps_embeds, ages_embeds),
            dim=-1,
        )

        inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        embeddings = inputs_embeds + token_type_embeds
        embeddings += visit_order_embeds
        embeddings += visit_segments_embeds

        embeddings = self.dropout(embeddings)

        return self.LayerNorm(embeddings)
    
    # def forward(
    #     self,
    #     data: Optional[torch.Tensor] = None,          # Replaces input_ids
    #     times: Optional[torch.Tensor] = None,         # Replaces time_stamps
    #     static: Optional[torch.Tensor] = None,        # Replaces ages and visit_segments
    #     mask: Optional[torch.Tensor] = None,          # Replaces token_type_ids_batch
    #     delta: Optional[torch.Tensor] = None,         # Replaces visit_orders
    #     labels: Optional[torch.Tensor] = None,        # Optional, only included to match current variables
    # ) -> Any:
    #     # Map your variables to the original expected ones
    #     # Step 1: Process `data` (time-series clinical variables)
    #     # Assume `data` has shape [batch_size, time_steps, num_features]
    #     # Use a linear layer to project continuous features into embedding space
    #     if self.scale_back_concat_layer.in_features != data.shape[-1]:
    #         input_dim = data.shape[-1]
    #         self.scale_back_concat_layer = nn.Linear(input_dim, self.hidden_size).to(data.device)

    #     # Apply the linear layer
    #     data_embeds = self.scale_back_concat_layer(data)
    #     # data_embeds = self.scale_back_concat_layer(data)  # Linear projection

    #     # Step 2: Process `times` (time since admission)
    #     # `times` has shape [batch_size, time_steps]
    #     time_embeds = self.time_embeddings(times)  # Temporal embeddings

    #     # Step 3: Process `static` features (e.g., demographic data like age, gender)
    #     # Assume `static` has shape [batch_size, num_static_features]
    #     # Embed static features using `visit_segment_embeddings`
    #     if static is not None:
    #         static = static.long()  # Cast to LongTensor
    #     offset = abs(static.min().item())
    #     static = static + offset  # Shift all values to be non-negative
    #     static = torch.clamp(static, min=0, max=self.visit_order_size - 1)

    #     # print("Static tensor shape:", static.shape)
    #     # print("Static tensor max value:", static.max().item())
    #     # print("Static tensor min value:", static.min().item())

    #     static_embeds = self.visit_segment_embeddings(static)

    #     # Step 4: Process `delta` (time gaps for sequence modeling)
    #     # `delta` has shape [batch_size, time_steps]
    #     visit_order_embeds = self.visit_order_embeddings(delta)

    #     # Step 5: Process `mask` as token type embeddings
    #     # `mask` has shape [batch_size, time_steps]
    #     token_type_embeds = self.token_type_embeddings(mask)

    #     # Step 6: Combine embeddings
    #     # Concatenate `data_embeds`, `time_embeds`, and expanded `static_embeds`
    #     # Expand static embeddings to match the sequence length for concatenation
    #     static_embeds = static_embeds.unsqueeze(1).expand(-1, data_embeds.size(1), -1)
    #     inputs_embeds = torch.cat([data_embeds, time_embeds.unsqueeze(-1), static_embeds], dim=-1)

    #     # Step 7: Scale back using linear transformation and add other embeddings
    #     inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
    #     embeddings = (
    #         inputs_embeds
    #         + token_type_embeds
    #         + visit_order_embeds
    #     )

    #     # Step 8: Normalize and apply dropout
    #     embeddings = self.LayerNorm(embeddings)
    #     embeddings = self.dropout(embeddings)

    #     return embeddings
    
        # Another solution(not work)
        # # Step 1: Process `data` (time-series clinical variables)
        # # `data` has shape [batch_size, time_steps, num_features]
        # data_embeds = self.word_embeddings(data)  # Each feature embedded

        # # Step 2: Process `times` (time since admission)
        # # `times` has shape [batch_size, time_steps]
        # time_embeds = self.time_embeddings(times)  # Embed scalar time values

        # # Step 3: Process `static` features (e.g., demographic data like age, gender)
        # # Assume `static` has shape [batch_size, num_static_features]
        # static_embeds = self.visit_segment_embeddings(static)  # Static embeddings

        # # Step 4: Process `delta` (time gaps for sequence modeling)
        # visit_order_embeds = self.visit_order_embeddings(delta)  # Delta embeddings

        # # Step 5: Process `mask` as token type embeddings
        # token_type_embeds = self.token_type_embeddings(mask)

        # # Step 6: Combine embeddings (concatenate along last dimension)
        # # Concatenate `data_embeds`, `time_embeds`, and `static_embeds`
        # inputs_embeds = torch.cat(
        #     [data_embeds, time_embeds.unsqueeze(-1), static_embeds.unsqueeze(1)], dim=-1
        # )  # Align dimensions for concatenation

        # # Step 7: Scale back using linear transformation and add other embeddings
        # inputs_embeds = self.tanh(self.scale_back_concat_layer(inputs_embeds))
        # embeddings = (
        #     inputs_embeds
        #     + token_type_embeds
        #     + visit_order_embeds
        # )

        # # Step 8: Normalize and apply dropout
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)

        # return embeddings
   

