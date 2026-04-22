"""
AgentBus – central message broker that routes messages to registered agents.
"""
from __future__ import annotations
import logging
from typing import Dict
from .base_agent import BaseAgent, Message

logger = logging.getLogger(__name__)


class AgentBus:
    """Routes Message objects to registered agents."""

    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        self._agents[agent.name] = agent
        logger.info("AgentBus: registered agent '%s'", agent.name)

    def send(self, message: Message) -> Message | None:
        """Deliver a message to the named receiver agent synchronously."""
        agent = self._agents.get(message.receiver)
        if agent is None:
            logger.error("AgentBus: no agent named '%s'", message.receiver)
            return Message(
                sender='bus',
                receiver=message.sender,
                msg_type='error',
                payload={'error': f"Agent '{message.receiver}' not found"},
            )
        logger.debug("AgentBus: routing %s -> %s [%s]", message.sender, message.receiver, message.msg_type)
        return agent.handle(message)

    def agents(self) -> list[str]:
        return list(self._agents.keys())
