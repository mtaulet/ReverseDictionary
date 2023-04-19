from transformers import BertTokenizer, BertForMaskedLM
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as F
from tqdm import tqdm
import torch
import numpy as np
import time

from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler
from src.data import CustomDataset, dataset_tokenized, get_wordvecs
from src.models.model_1 import Model1


def weighted_mse_loss(y_hat, y, weights):
    return (weights * ((y_hat - y) ** 2).mean(1)).mean()


class Experiment3:
    def __init__(self, checkpoint_dir, batch_size=16):
        self.model = Model1()
        self.dataset = CustomDataset(*dataset_tokenized())
        self.checkpoint_dir = checkpoint_dir

        # Set up dataloaders
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(224))
        self.train_dataloader = DataLoader(train_dataset,
                                           sampler=RandomSampler(train_dataset,
                                               generator=torch.Generator().manual_seed(224)),
                                           batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset,
                                         sampler=SequentialSampler(val_dataset),
                                         batch_size=batch_size)
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # Loss function - In Experiment3, we want to use a weighted MSE loss.
        self.loss_fn = weighted_mse_loss
        print(f"=> Set up experiment with model {self.model.__class__} and batch size {batch_size}.")
        print(f"=> Checkpoints saving to {self.checkpoint_dir}")
        print(f"=> Model running on device {self.device}")

    def set_train(self, hold_backend_constant=True):
        self.model.train()
        for name, param in self.model.bert_backend.named_parameters():
            param.requires_grad = not hold_backend_constant

    def evaluate(self, dataloader):
        top_count = {"top_1": 0,
                     "top_10": 0,
                     "top_100": 0}
        total_eval_loss = 0.0
        wordvecs = get_wordvecs()
        self.model.to(self.device)
        self.model.eval()

        t0 = time.time()
        print("  INFO: Evaluate:")
        for i_, batch in enumerate(tqdm(dataloader, ascii=True)):
            tb = time.time()
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                output = self.model(b_input_ids, b_input_mask)

            output = output.cpu().numpy().astype(np.float32)
            b_labels = b_labels.cpu().numpy().astype(np.float32)
            weights = []
            for i in range(len(output)):
                top_100 = wordvecs.most_similar(positive=[output[i]], topn=100)
                top_100_words = [t[0] for t in top_100]
                top_1_word = top_100_words[0]
                actual = wordvecs.most_similar(positive=[b_labels[i]], topn=1)[0][0]

                top_count["top_1"] += int(actual == top_1_word)
                top_count["top_10"] += int(actual in top_100_words[:10])
                top_count["top_100"] += int(actual in top_100_words)
                top_100_words.append(actual)
                idx = top_100_words.index(actual)
                weight = max((idx+1), 100) / 100.0
                weights.append(weight)
            weights = torch.Tensor(weights)
            output = torch.Tensor(output)
            b_labels = torch.Tensor(b_labels)
            total_eval_loss += self.loss_fn(output, b_labels, weights)
            # print(f"  INFO: Evaluate batch time ({i_ + 1}/{len(dataloader)}): {time.time() - tb}")
        # Get averages
        avg_top1_acc = top_count["top_1"] / len(dataloader.dataset)
        avg_top10_acc = top_count["top_10"] / len(dataloader.dataset)
        avg_top100_acc = top_count["top_100"] / len(dataloader.dataset)
        avg_loss = total_eval_loss / len(dataloader)
        print(f"  INFO: Evaluate total time: {time.time() - t0}")
        return {
            "top1_acc": avg_top1_acc,
            "top10_acc": avg_top10_acc,
            "top100_acc": avg_top100_acc,
            "loss": avg_loss,
        }


    def train(self, n_epochs=4, fine_tune_epochs=1, evaluate_every=5, gradient_clip=None):
        assert fine_tune_epochs <= n_epochs
        self.model.to(self.device)
        # Note: AdamW is from the huggingface library for WeightDecay fix
        optimizer = AdamW(self.model.parameters(),
                          lr=1e-5,    # args.learning_rate
                          eps=1e-8)   # args.adam_epsilon
        total_steps = len(self.train_dataloader) * n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        wordvecs = get_wordvecs()
        for epoch in range(n_epochs):
            print(f"======= Epoch {epoch+1} / {n_epochs} =======")
            epoch_loss = 0.0
            # Only fine tune the BERT backend in the last K epochs
            fine_tune = (epoch + fine_tune_epochs) >= n_epochs
            if fine_tune:
                print("  INFO: Fine tuning BERT backend this epoch.")
            self.set_train(hold_backend_constant=(not fine_tune))
            for step, batch in enumerate(tqdm(self.train_dataloader, ascii=True)):
                """
                if (step + 1) % 50 == 0:
                    print(f"    Batch {step+1} of {len(self.train_dataloader)}")
                """

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                output = self.model(b_input_ids, b_input_mask)

                # Get weights. The weight for any (x, y, y_hat) is equal to max(rank, 100) / 100. i.e rank 1 is scaled
                # by 0.01, but rank 100+ is not scaled at all. This makes it so that better similarities (i.e. rank 1)
                # is updated less, but worse ranks are updated more.
                output_cpy = output.detach().clone().cpu().numpy().astype(np.float32)
                label_cpy = b_labels.cpu().numpy().astype(np.float32)
                weights = []
                for i in range(len(output)):
                    top_100 = wordvecs.most_similar(positive=[output_cpy[i]], topn=100)
                    actual = wordvecs.most_similar(positive=[label_cpy[i]], topn=1)[0][0]

                    # Get the index of actual from top_100. Note that list.index will error if x is not found in the
                    # list, so we can use a hack where we append `actual` to the top_100 list. This is handled nicely by
                    # our weight function.
                    top_100_words = [t[0] for t in top_100]
                    top_100_words.append(actual)
                    idx = top_100_words.index(actual)
                    weight = max((idx+1), 100) / 100.0
                    weights.append(weight)
                weights = torch.Tensor(weights).to(self.device)
                batch_loss = self.loss_fn(output, b_labels.float(), weights)
                epoch_loss += batch_loss.item()
                batch_loss.double().backward()

                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                optimizer.step()
                scheduler.step()
            torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/params.pt.{epoch+1}")
            epoch_loss /= len(self.train_dataloader)
            print(f"  Epoch loss: {epoch_loss}")

            if (epoch + 1) % evaluate_every == 0:
                validation_stats = self.evaluate(self.val_dataloader)
                print(f"  Validation Top 1 Accuracy: {validation_stats['top1_acc']}")
                print(f"  Validation Top 10 Accuracy: {validation_stats['top10_acc']}")
                print(f"  Validation Top 100 Accuracy: {validation_stats['top100_acc']}")
                print(f"  Validation Loss: {validation_stats['loss']}")
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/params.pt.final")

