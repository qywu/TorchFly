import inspect
import logging

from .callback import Callback

logger = logging.getLogger(__name__)


def _is_event_handler(member) -> bool:
    return inspect.ismethod(member) and hasattr(member, "_event") and hasattr(member, "_priority")

class EventHandler(NamedTuple):
    name: str
    callback: Callback
    handler: Callable[[TrainerBase], None]
    priority: int


class CallbackHandler():
    def __init__(self, trainer=None, callbacks=None, verbose=False):
        """
        Args:
            verbose : bool, optional (default = False) Used for debugging.
        """
        self.trainer = trainer

        self._callbacks: Dict[str, List[EventHandler]] = defaultdict(list)

        for callback in callbacks:
            self.add_callback(callback)

        self.verbose = verbose

    def callbacks(self) -> List[Callback]:
        """
        Returns the callbacks associated with this handler.
        Each callback may be registered under multiple events,
        but we make sure to only return it once. If `typ` is specified,
        only returns callbacks of that type.
        """
        return list({callback.callback for callback_list in self._callbacks.values() for callback in callback_list})

    def add_callback(self, callback: Callback) -> None:
        for name, method in inspect.getmembers(callback, _is_event_handler):
            event = getattr(method, "_event")
            priority = getattr(method, "_priority")
            self._callbacks[event].append(EventHandler(name, callback, method, priority))
            self._callbacks[event].sort(key=lambda eh: eh.priority)

            self._callbacks_by_type[type(callback)].append(callback)

    def fire_event(self, event: str) -> None:
        """
        Runs every callback registered for the provided event,
        ordered by their priorities.
        """
        for event_handler in self._callbacks.get(event, []):
            if self.verbose:
                logger.debug(f"event {event} -> {event_handler.name}")
            event_handler.handler(self.trainer)