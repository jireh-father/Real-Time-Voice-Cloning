from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch
import random
import numpy as np


def sync(device: torch.device):
    # FIXME
    return
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool, num_workers=8, prefix_list_str=None, num_epochs=100):
    # Create a dataset and a dataloader
    random.seed(1)
    if prefix_list_str is None:
        train_speaker_dirs = [f for f in clean_data_root.glob("*") if f.is_dir()]
        random.shuffle(train_speaker_dirs)
        cut_idx = int(len(train_speaker_dirs) * 0.1)
        test_speaker_dirs = train_speaker_dirs[:cut_idx]
        train_speaker_dirs = train_speaker_dirs[cut_idx:]
    else:
        train_speaker_dirs = []
        test_speaker_dirs = []
        prefix_list = prefix_list_str.split(",")
        for prefix_str in prefix_list:
            tmp_dirs = [f for f in clean_data_root.glob(prefix_str + "*") if f.is_dir()]
            random.shuffle(tmp_dirs)
            cut_idx = int(len(tmp_dirs) * 0.1)
            test_speaker_dirs += tmp_dirs[:cut_idx]
            train_speaker_dirs += tmp_dirs[cut_idx:]

    train_dataset = SpeakerVerificationDataset(train_speaker_dirs)
    test_dataset = SpeakerVerificationDataset(test_speaker_dirs)
    train_loader = SpeakerVerificationDataLoader(
        train_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=num_workers,
    )

    test_loader = SpeakerVerificationDataLoader(
        test_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=num_workers,
    )

    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cpu")

    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    init_step = 1

    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(train_dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})

    # Training loop
    profiler = Profiler(summarize_every=10, disabled=False)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = []
        total_eer = []
        for step, speaker_batch in enumerate(train_loader, init_step):
            profiler.tick("Blocking, waiting for batch (threaded)")

            # Forward pass
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            sync(device)
            profiler.tick("Data to %s" % device)
            embeds = model(inputs)
            sync(device)
            profiler.tick("Forward pass")
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            total_loss.append(loss)
            total_eer.append(eer)
            sync(loss_device)
            profiler.tick("Loss")

            # Backward pass
            model.zero_grad()
            loss.backward()
            profiler.tick("Backward pass")
            model.do_gradient_ops()
            optimizer.step()
            profiler.tick("Parameter update")

            # Update visualizations
            # learning_rate = optimizer.param_groups[0]["lr"]
            vis.update(loss.item(), eer, step)

            # Draw projections and save them to the backup folder
            if umap_every != 0 and step % umap_every == 0:
                print("Drawing and saving projections (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
                embeds = embeds.detach().cpu().numpy()
                vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
                vis.save()

            # # Overwrite the latest version of the model
            # if save_every != 0 and step % save_every == 0:
            #     print("Saving the model (step %d)" % step)
            #     torch.save({
            #         "step": step + 1,
            #         "model_state": model.state_dict(),
            #         "optimizer_state": optimizer.state_dict(),
            #     }, state_fpath)
            #
            # # Make a backup
            # if backup_every != 0 and step % backup_every == 0:
            #     print("Making a backup (step %d)" % step)
            #     backup_dir.mkdir(exist_ok=True)
            #     backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            #     torch.save({
            #         "step": step + 1,
            #         "model_state": model.state_dict(),
            #         "optimizer_state": optimizer.state_dict(),
            #     }, backup_fpath)

            profiler.tick("Extras (visualizations, saving)")
        print(
            "epoch %d: avg train loss: %4f, avg train eer: %4f" % (
            epoch, np.array(total_loss).mean(), np.array(total_eer).mean()))

        print("Saving the model (epoch %d)" % epoch)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, models_dir.joinpath(run_id + "_%d_epoch.pt" % epoch))

        total_loss = []
        total_eer = []
        model.eval()
        for step, speaker_batch in enumerate(test_loader):
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            embeds = model(inputs)
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            total_loss.append(loss)
            total_eer.append(eer)
        print(
            "epoch %d: avg val loss: %4f, avg val eer: %4f" % (epoch, np.array(total_loss).mean(), np.array(total_eer).mean()))
