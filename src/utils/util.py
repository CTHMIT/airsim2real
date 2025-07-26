
import sys
from pathlib import Path
from queue import Queue
from typing import Optional,Callable
from pydantic import BaseModel, Field
from utils.logger import LOGGER  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

class EventBus:
    """
    Event bus for decoupled communication between components
    Replaces global queues with a publish-subscribe pattern
    """

    def __init__(self):
        self._subscribers = {}
        self._queues = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type with a callback function"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data=None):
        """Publish an event to all subscribers"""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    LOGGER.error(f"Error in subscriber callback for {event_type}: {e}")

    def create_queue(self, queue_name: str, maxsize: int = 0) -> Queue:
        """Create a named queue that can be accessed by components"""
        queue: Queue = Queue(maxsize=maxsize)
        self._queues[queue_name] = queue
        return queue

    def get_queue(self, queue_name: str) -> Optional[Queue]:
        """Get a queue by name"""
        return self._queues.get(queue_name)


# Global event bus instance
event_bus = EventBus()