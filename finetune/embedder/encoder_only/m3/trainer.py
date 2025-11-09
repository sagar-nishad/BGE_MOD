# import os
# import torch
# import logging
# from typing import Optional

# from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

# logger = logging.getLogger(__name__)


# class EncoderOnlyEmbedderM3Trainer(AbsEmbedderTrainer):
#     """
#     Trainer class for M3.
#     """
#     def _save(self, output_dir: Optional[str] = None, state_dict=None):
#         """Save the model to directory.

#         Args:
#             output_dir (Optional[str], optional): Output directory to save the model. Defaults to ``None``.

#         Raises:
#             NotImplementedError
#         """
#         output_dir = output_dir if output_dir is not None else self.args.output_dir
#         os.makedirs(output_dir, exist_ok=True)
#         logger.info("Saving model checkpoint to %s", output_dir)
#         # Save a trained model and configuration using `save_pretrained()`.
#         # They can then be reloaded using `from_pretrained()`
#         if not hasattr(self.model, 'save'):
#             raise NotImplementedError(
#                 f'MODEL {self.model.__class__.__name__} '
#                 f'does not support save interface')
#         else:
#             self.model.save(output_dir)
#         if self.tokenizer is not None and self.is_world_process_zero():
#             self.tokenizer.save_pretrained(output_dir)

#         torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

#         # save the checkpoint for sentence-transformers library
#         # if self.is_world_process_zero():
#         #     save_ckpt_for_sentence_transformers(output_dir,
#         #                                         pooling_mode=self.args.sentence_pooling_method,
#         #                                         normlized=self.args.normlized)


# ! new code gemini

import os
from torch import nn
import torch
import logging
from typing import Optional, Dict, Any  # <-- Import Dict and Any
from typing import Optional, Dict, Any, Tuple, Union

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderM3Trainer(AbsEmbedderTrainer):
    """
    Trainer class for M3.
    """
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction_step to force loss computation.
        
        The default Trainer.prediction_step checks for "labels" in the
        inputs, but our model computes loss from "queries" and "passages".
        This override bypasses that check and calls self.compute_loss directly.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                # We call compute_loss directly
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            
            # We don't have logits or labels in the traditional sense
            logits = None
            labels = None

        return (loss, logits, labels)
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, Any]:
        """
        Runs evaluation and returns metrics.
        """
        # This line calls the parent Trainer's evaluate method,
        # which correctly computes the loss and returns it in a
        # dictionary with the key 'eval_loss'.
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        return metrics
    # --- END OF NEW METHOD ---


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model to directory.

        Args:
            output_dir (Optional[str], optional): Output directory to save the model. Defaults to ``None``.

        Raises:
            NotImplementedError
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # ... (rest of your save method)