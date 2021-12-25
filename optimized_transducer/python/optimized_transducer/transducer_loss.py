# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)

import _optimized_transducer
import torch


class TransducerLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Args:
          logits:
            A 2-D tensor of shape (sum_all_TU, vocab_size). Suppose you have a
            batch containing two utterances, in a normal setup, the output of
            the joint network is (2, max(T1, T2), max(U1, U2), vocab_size);
            however, in optimized transducer, we expect that you first convert
            (T1, U1, vocab_size) and (T2, U2, vocab_size) to (T1*U1, vocab_size)
            and (T2, U2, vocab_size), respectively; and then concatenate
            (T1*U1, vocab_size) and (T2*U2, vocab_size) to get
            (T1*U1 + T2*U2, vocab_size).

            Caution:
              You need to do the above transformation at the input of the joint
              network to save memory, not at the output of the joint network.
              In other words, you have to modify your joint network to accept
              2-D tensors as inputs and also produce a 2-D tensor as output.
              This is feasible as a joint network usually contains only
              `nn.Linear` and pointwise activation layers, where frames are
              processed independently.

            Caution:
              Its dtype has to be torch.float32 and it has to be contiguous
              in memory.

            Caution:
              If it requires grad, then it shares the same memory with its
              gradient. After the call, its original value is overwritten
              with its gradient.

            Hint:
              `logits` contains unnormalized probabilities and it is usually the
              output of some `nn.Linear` layer.
          targets:
            A 2-D tensor of shape (batch_size, num_tokens) containing  the
            tokens in each utterance.

            Caution:
              Its dtype has to be torch.int32 and has to be contiguous in
              memory. Must be on the same device as `logits`.
          logit_lengths:
            A 1-D tensor of shape (batch_size,) containing the number of output
            frames from the encoder, i.e., `T_i`s

            Caution:
              Its dtype has to be torch.int32 and has to be contiguous in
              memory. Must be on the same device as `logits`.
          target_lengths:
            A 1-D tensor of shape (batch_size, ) containing the number of
            tokens in each utterance before padding.

            Caution:
              It does not count the blank symbol.

            Caution:
              Its dtype has to be torch.int32 and has to be contiguous in
              memory. Must be on the same device as `logits`.
          blank:
            The ID of the blank symbol. Must be 0 <= blank <= logits.size(1)-1.
          reduction:
            Supported values are:

                - "mean". Return a tensor of shape (1,) containing the average
                          loss over all utterances in the batch.
                - "sum".  Return a tensor of shape (1,) containing the sum of
                          the loss over all utterances in the batch.
            Caution:
              We don't support "none" as it increases the difficulty to compute
              the gradient.
        Returns:
          Return a tensor containing the losses. See the documentation above for
          `reduction`.
        """
        assert reduction in ("mean", "sum")

        scores, grad = _optimized_transducer.compute_transducer_loss(
            logits=logits,
            targets=targets,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            blank=blank,
        )

        loss = -1 * scores

        if reduction == "mean":
            loss = loss.mean()
            grad /= logit_lengths.size(0)
        elif reduction == "sum":
            loss = loss.sum()

        ctx.grad = grad

        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        assert loss_grad.numel() == 1, loss_grad.numel()
        return (
            ctx.grad.mul_(loss_grad),  # logits
            None,  # targets,
            None,  # logit_lengths,
            None,  # target_lengths
            None,  # blank
            None,  # reduction
        )


class TransducerLoss(torch.nn.Module):
    def __init__(self, blank: int, reduction: str = "mean"):
        """
        Args:
          blank:
            The ID for the blank symbol.
          reduction:
            Supported values are:

                - "mean". Return a tensor of shape (1,) containing the average
                          loss over all utterances in the batch.
                - "sum".  Return a tensor of shape (1,) containing the sum of
                          the loss over all utterances in the batch.
        """
        self.blank = blank
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        logit_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          logits:
            A 2-D tensor of shape (sum_all_TU, vocab_size). Suppose you have a
            batch containing two utterances, in a normal setup, the output of
            the joint network is (2, max(T1, T2), max(U1, U2), vocab_size);
            however, in optimized transducer, we expect that you first convert
            (T1, U1, vocab_size) and (T2, U2, vocab_size) to (T1*U1, vocab_size)
            and (T2, U2, vocab_size), respectively; and then concatenate
            (T1*U1, vocab_size) and (T2*U2, vocab_size) to get
            (T1*U1 + T2*U2, vocab_size).

            Caution:
              You need to do the above transformation at the input of the joint
              network to save memory, not at the output of the joint network.
              In other words, you have to modify your joint network to accept
              2-D tensors as inputs and also produce a 2-D tensor as output.
              This is feasible as a joint network usually contains only
              `nn.Linear` and pointwise activation layers, where frames are
              processed independently.

            Caution:
              Its dtype has to be torch.float32 and it has to be contiguous
              in memory.

            Caution:
              If it requires grad, then it shares the same memory with its
              gradient. After the call, its original value is overwritten
              with its gradient.

            Hint:
              `logits` contains unnormalized probabilities and it is usually the
              output of some `nn.Linear` layer.
          targets:
            A 2-D tensor of shape (batch_size, num_tokens) containing  the
            tokens in each utterance.

            Caution:
              Its dtype has to be torch.int32 and has to be contiguous in
              memory. Must be on the same device as `logits`.
          logit_lengths:
            A 1-D tensor of shape (batch_size,) containing the number of output
            frames from the encoder, i.e., `T_i`s

            Caution:
              Its dtype has to be torch.int32 and has to be contiguous in
              memory. Must be on the same device as `logits`.
          target_lengths:
            A 1-D tensor of shape (batch_size, ) containing the number of
            tokens in each utterance before padding.

            Caution:
              It does not count the blank symbol.

            Caution:
              Its dtype has to be torch.int32 and has to be contiguous in
              memory. Must be on the same device as `logits`.
        Returns:
          Return a tensor containing the losses. See the documentation above for
          `self.reduction`.
        """
        return TransducerLossFunction.apply(
            logits,
            targets,
            logit_lengths,
            target_lengths,
            self.blank,
            self.reduction,
        )


def transducer_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Args:
      logits:
        A 2-D tensor of shape (sum_all_TU, vocab_size). Suppose you have a
        batch containing two utterances, in a normal setup, the output of
        the joint network is (2, max(T1, T2), max(U1, U2), vocab_size);
        however, in optimized transducer, we expect that you first convert
        (T1, U1, vocab_size) and (T2, U2, vocab_size) to (T1*U1, vocab_size)
        and (T2, U2, vocab_size), respectively; and then concatenate
        (T1*U1, vocab_size) and (T2*U2, vocab_size) to get
        (T1*U1 + T2*U2, vocab_size).

        Caution:
          You need to do the above transformation at the input of the joint
          network to save memory, not at the output of the joint network.
          In other words, you have to modify your joint network to accept
          2-D tensors as inputs and also produce a 2-D tensor as output.
          This is feasible as a joint network usually contains only
          `nn.Linear` and pointwise activation layers, where frames are
          processed independently.

        Caution:
          Its dtype has to be torch.float32 and it has to be contiguous
          in memory.

        Caution:
          If it requires grad, then it shares the same memory with its
          gradient. After the call, its original value is overwritten
          with its gradient.

        Hint:
          `logits` contains unnormalized probabilities and it is usually the
          output of some `nn.Linear` layer.
      targets:
        A 2-D tensor of shape (batch_size, num_tokens) containing  the
        tokens in each utterance.

        Caution:
          Its dtype has to be torch.int32 and has to be contiguous in
          memory. Must be on the same device as `logits`.
      logit_lengths:
        A 1-D tensor of shape (batch_size,) containing the number of output
        frames from the encoder, i.e., `T_i`s

        Caution:
          Its dtype has to be torch.int32 and has to be contiguous in
          memory. Must be on the same device as `logits`.
      target_lengths:
        A 1-D tensor of shape (batch_size, ) containing the number of
        tokens in each utterance before padding.

        Caution:
          It does not count the blank symbol.

        Caution:
          Its dtype has to be torch.int32 and has to be contiguous in
          memory. Must be on the same device as `logits`.
      blank:
        The ID of the blank symbol. Must be 0 <= blank <= logits.size(1)-1.
      reduction:
        Supported values are:

            - "mean". Return a tensor of shape (1,) containing the average
                      loss over all utterances in the batch.
            - "sum".  Return a tensor of shape (1,) containing the sum of
                      the loss over all utterances in the batch.
    Returns:
      Return a tensor containing the losses. See the documentation above for
      `reduction`.
    """
    return TransducerLossFunction.apply(
        logits,
        targets,
        logit_lengths,
        target_lengths,
        blank,
        reduction,
    )
