"""
Base Agent class – defines the contract every agent must follow.
All agents communicate via the agent_messages table (async) or
direct method calls (sync, for simplicity in demo mode).
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Message:
    """Lightweight message container passed between agents."""

    def __init__(self, sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]):
        self.sender = sender
        self.receiver = receiver
        self.msg_type = msg_type
        self.payload = payload
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'msg_type': self.msg_type,
            'payload': self.payload,
            'timestamp': self.timestamp,
        }

    def __repr__(self) -> str:
        return f"Message({self.sender} -> {self.receiver}: {self.msg_type})"


class BaseAgent:
    """
    Abstract base class for all system agents.

    Sub-classes must implement `handle(message)` to process
    incoming messages and return a response Message (or None).
    """

    def __init__(self, name: str, db_conn):
        self.name = name
        self.db = db_conn
        self._handlers: Dict[str, Any] = {}
        self._register_handlers()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------
    def _register_handlers(self):
        """Subclasses override this to register msg_type -> handler."""

    def register(self, msg_type: str, handler):
        self._handlers[msg_type] = handler

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------
    def handle(self, message: Message) -> Message | None:
        handler = self._handlers.get(message.msg_type)
        if handler is None:
            logger.warning("%s: no handler for msg_type '%s'", self.name, message.msg_type)
            return self._reply(message, 'error', {'error': f"Unknown message type: {message.msg_type}"})
        try:
            result = handler(message)
            return result
        except Exception as exc:
            logger.exception("%s: error handling %s", self.name, message.msg_type)
            return self._reply(message, 'error', {'error': str(exc)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reply(self, original: Message, msg_type: str, payload: Dict[str, Any]) -> Message:
        return Message(
            sender=self.name,
            receiver=original.sender,
            msg_type=msg_type,
            payload=payload,
        )

    def _persist_message(self, message: Message):
        """Persist a message to the agent_messages table for audit."""
        self.db.execute(
            "INSERT INTO agent_messages(sender, receiver, msg_type, payload) VALUES (?,?,?,?)",
            (message.sender, message.receiver, message.msg_type, json.dumps(message.payload, ensure_ascii=False)),
        )
        self.db.commit()
