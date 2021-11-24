import torch

from ignite.engine import Events

import datetime

import numpy as np

from pathlib import Path

from src.trainer import Trainer, MyEngine
from src.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class EngineForBart(MyEngine):

    def __init__(
        self, 
        func, 
        model, 
        crit, 
        optimizer, 
        scheduler, 
        config,
    ):
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):
        ## You have to reset the gradients of all model parameters
        ## before to take another step in gradient descent.
        engine.model.train() ## because we assign model as class variable, we can easily access to it
        engine.optimizer.zero_grad()

        ## Unpack with lazy loading.
        input_ids              = mini_batch["input_ids"].to(engine.device)
        attention_mask         = mini_batch["attention_mask"].to(engine.device)
        decoder_input_ids      = mini_batch["decoder_input_ids"].to(engine.device)
        decoder_attention_mask = mini_batch["decoder_attention_mask"].to(engine.device)
        labels                 = mini_batch["labels"].to(engine.device)
        ## |input_ids|              = (batch_size, variable_max_length !< inp_max_length)
        ## |attention_mask|         = (batch_size, variable_max_length !< inp_max_length)
        ## |decoder_input_ids|      = (batch_size, variable_max_length !< tar_max_length)
        ## |decoder_attention_mask| = (batch_size, variable_max_length !< tar_max_length)
        ## |labels|                 = (batch_size, variable_max_length !< tar_max_length)

        ## We don't need to slice sentences by 'config.*_max_length'
        ## baecause the tokenizer will do that in 'collate_fn'.
        # x = x[:, :engine.config.max_length]

        with torch.cuda.amp.autocast():
            ## Take feed-forward
            output = engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True,
            )
            ## type(output) == Seq2SeqLMOutput
            ## A Seq2SeqLMOutput or a tuple of torch.FloatTensor (if return_dict=False 
            ## is passed or when config.return_dict=False) comprising various elements 
            ## depending on the configuration (BartConfig) and inputs.
            loss = output[0]

        ## If we are using gpu, not cpu,
        if engine.config.gpu_id >= 0:
            engine.scaler.scale(loss).backward()
        else:
            loss.backward()

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        ## In order to avoid gradient exploding, we apply gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        ## Take a step of gradient descent.
        if engine.config.gpu_id >= 0:
            ## Use caler instead of engine.optimizer.step() if using GPU.
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            engine.optimizer.step()

        if engine.scheduler != None:
            engine.scheduler.step()

        ## word_count = int(y.tgt[1].sum())
        ## loss = float(loss / word_count)
        loss = loss.cpu().detach().numpy()
        ppl = np.exp(loss)

        return {
            "loss": loss,
            "ppl": ppl,
            "|param|": p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            "|g_param|": g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            ## Unpack with lazy loading.
            input_ids              = mini_batch["input_ids"].to(engine.device)
            attention_mask         = mini_batch["attention_mask"].to(engine.device)
            decoder_input_ids      = mini_batch["decoder_input_ids"].to(engine.device)
            decoder_attention_mask = mini_batch["decoder_attention_mask"].to(engine.device)
            labels                 = mini_batch["labels"].to(engine.device)
            ## |input_ids|              = (batch_size, variable_max_length !< inp_max_length)
            ## |attention_mask|         = (batch_size, variable_max_length !< inp_max_length)
            ## |decoder_input_ids|      = (batch_size, variable_max_length !< tar_max_length)
            ## |decoder_attention_mask| = (batch_size, variable_max_length !< tar_max_length)
            ## |labels|                 = (batch_size, variable_max_length !< tar_max_length)

            ## We don't need to slice sentences by 'config.*_max_length'
            ## baecause the tokenizer will do that in 'collate_fn'.
            # x = x[:, :engine.config.max_length]

            with torch.cuda.amp.autocast():
                ## Take feed-forward
                output = engine.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                ## type(output) == Seq2SeqLMOutput
                ## A Seq2SeqLMOutput or a tuple of torch.FloatTensor (if return_dict=False 
                ## is passed or when config.return_dict=False) comprising various elements 
                ## depending on the configuration (BartConfig) and inputs.
                loss = output[0]

            ## word_count = int(y.tgt[1].sum())
            ## loss = float(loss / word_count)
            loss = loss.cpu().detach().numpy()
            ppl = np.exp(loss)

            return {
                "loss": loss,
                "ppl": ppl,
            }


    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    
    @staticmethod
    def save_model(engine, train_engine, model_dir, config):
        avg_train_loss = train_engine.state.metrics["loss"]
        avg_valid_loss = engine.state.metrics["loss"]

        model_fname = Path(".".join([
            config.model_fpath,                                    ## user-entered hyper-params
            "%02d" % train_engine.state.epoch,                      ## current epoch
            "%.2f-%.2f" % (avg_train_loss, np.exp(avg_train_loss)), ## train assets
            "%.2f-%.2f" % (avg_valid_loss, np.exp(avg_valid_loss)), ## valid assets
            "pth",                                                  ## extension
        ]))

        ## Unlike other tasks, we need to save current model, not best model.
        torch.save({
            "bart": engine.model.state_dict(),
            "config": config,
        }, model_dir / model_fname)


class BartTrainer(Trainer):

    def __init__(
        self, 
        config,
    ):
        self.config = config
        
        ## Set a filename for model of last epoch.
        ## We need to put every information to filename, as much as possible.
        self.model_dir = Path(config.ckpt, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        model, 
        crit, 
        optimizer, 
        scheduler,
        train_loader, 
        valid_loader,
    ):
        train_engine = EngineForBart(
            EngineForBart.train,
            model, 
            crit, 
            optimizer, 
            scheduler, 
            self.config,
        )
        validation_engine = EngineForBart(
            EngineForBart.validate,
            model, 
            crit, 
            optimizer=None, ## no need to throw optimizer
            scheduler=None, ## no need to throw scheduler
            config=self.config,
        )

        EngineForBart.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose,
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,          ## event
            run_validation,                  ## function
            validation_engine, valid_loader, ## arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,          ## event
            EngineForBart.check_best,        ## function
        )
        ## Save models for each epochs.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForBart.save_model,
            train_engine,
            self.model_dir,
            self.config,
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        return model
