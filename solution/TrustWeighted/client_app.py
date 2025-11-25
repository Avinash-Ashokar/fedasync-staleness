
"""TrustWeighted: A Flower / PyTorch Lightning app (ClientApp)."""

from __future__ import annotations

import pytorch_lightning as pl
from datasets.utils.logging import disable_progress_bar
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from TrustWeighted.task import LitCifar, load_data

# Disable tqdm progress bars from HuggingFace/datasets to keep logs clean
disable_progress_bar()

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on the local partition.

    The server sends the current global model parameters as an ``ArrayRecord``
    under the key ``"arrays"``. We:

    * load the global weights into a fresh ``LitAutoEncoder``
    * train for ``max-epochs`` on the local data partition
    * report back the updated weights and training loss
    """

    # -------------------------------------------------------------------------
    # 1) Rebuild model and load global parameters
    # -------------------------------------------------------------------------
    model = LitCifar()
    global_arrays: ArrayRecord = msg.content["arrays"]  # type: ignore[index]
    model.load_state_dict(global_arrays.to_torch_state_dict())

    # -------------------------------------------------------------------------
    # 2) Load local data partition
    # -------------------------------------------------------------------------
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader, _ = load_data(partition_id, num_partitions)

    # -------------------------------------------------------------------------
    # 3) Run local training with PyTorch Lightning
    # -------------------------------------------------------------------------
    max_epochs = int(context.run_config.get("max-epochs", 1))
    base_round = int(context.run_config.get("server-round", 0))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

    # Retrieve averaged training loss from Lightning's logged metrics
    train_loss_tensor = trainer.callback_metrics.get("train_loss")
    train_loss = float(train_loss_tensor.item()) if train_loss_tensor is not None else 0.0

    # -------------------------------------------------------------------------
    # 4) Package updated model and metrics into a Message
    # -------------------------------------------------------------------------
    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
            {
                "train_loss": train_loss,
                # This key is used by FedAvg/AsyncTrustFedAvg as the aggregation weight
                "num-examples": len(trainloader.dataset),
                # Round in which this client received the global model
                "base-round": base_round,
            }
        )
    content = RecordDict({"arrays": arrays, "metrics": metrics})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local test data.

    The server sends the current global model parameters as ``"arrays"``.
    We evaluate on the local test partition and report a single ``MetricRecord``.
    """

    # Rebuild model and load the parameters to evaluate
    model = LitCifar()
    arrays: ArrayRecord = msg.content["arrays"]  # type: ignore[index]
    model.load_state_dict(arrays.to_torch_state_dict())

    # Load local test data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    _, _, testloader = load_data(partition_id, num_partitions)

    trainer = pl.Trainer(
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    test_results = trainer.test(model, dataloaders=testloader, verbose=False)

    test_loss = float(test_results[0].get("test_loss", 0.0)) if test_results else 0.0

    metrics = MetricRecord(
            {
                "test_loss": test_loss,
                "num-examples": len(testloader.dataset),
            }
        )
    content = RecordDict({"metrics": metrics})

    return Message(content=content, reply_to=msg)
