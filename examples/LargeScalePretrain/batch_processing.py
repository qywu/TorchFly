from torchfly.training.callbacks import Events, handle_event, Callback


class BatchHandler(Callback):

    @handle_event(Events.BATCH_BEGIN)
    def process_batch(self, trainer):
        breakpoint()
        pass