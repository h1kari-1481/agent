"""
Flask REST API – exposes all agent capabilities over HTTP.

Endpoints follow the pattern: POST /api/<agent>/<action>
Body is JSON and is forwarded as the message payload.
Authentication is via a simple session token stored in a header.
"""
from __future__ import annotations
import logging
from functools import wraps
from flask import Flask, request, jsonify, g

from ..models.database import init_db, get_connection
from ..agents.base_agent import Message
from ..agents.agent_bus import AgentBus
from ..agents.enterprise_agent import EnterpriseAgent
from ..agents.city_agent import CityAgent
from ..agents.province_agent import ProvinceAgent
from ..agents.system_agent import SystemAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --------------------------------------------------------------------------
# Initialise DB and agents
# --------------------------------------------------------------------------

init_db()
_conn = get_connection()
bus = AgentBus()
bus.register_agent(SystemAgent('system', _conn))
bus.register_agent(EnterpriseAgent('enterprise', _conn))
bus.register_agent(CityAgent('city', _conn))
bus.register_agent(ProvinceAgent('province', _conn))

# --------------------------------------------------------------------------
# Simple in-memory session store  {token -> {user_id, username, role, region}}
# --------------------------------------------------------------------------
_sessions: dict = {}


def _require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('X-Auth-Token', '')
        session = _sessions.get(token)
        if session is None:
            return jsonify({'error': '未登录或会话已过期'}), 401
        g.session = session
        return f(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

@app.route('/api/login', methods=['POST'])
def login():
    body = request.get_json(force=True) or {}
    msg = Message('api', 'system', 'login', body)
    resp = bus.send(msg)
    if resp.msg_type == 'login_success':
        import uuid
        token = str(uuid.uuid4())
        _sessions[token] = resp.payload
        return jsonify({'token': token, **resp.payload})
    return jsonify(resp.payload), 401


@app.route('/api/logout', methods=['POST'])
@_require_auth
def logout():
    token = request.headers.get('X-Auth-Token', '')
    _sessions.pop(token, None)
    return jsonify({'message': '已退出登录'})


# --------------------------------------------------------------------------
# Generic agent proxy endpoint
# --------------------------------------------------------------------------

@app.route('/api/<agent_name>/<action>', methods=['POST'])
@_require_auth
def agent_proxy(agent_name: str, action: str):
    body = request.get_json(force=True) or {}
    # Inject session context
    body.setdefault('user_id', g.session.get('user_id'))
    body.setdefault('region_code', g.session.get('region_code', ''))
    msg = Message('api', agent_name, action, body)
    resp = bus.send(msg)
    if resp is None:
        return jsonify({'error': 'No response from agent'}), 500
    status = 400 if resp.msg_type == 'error' else 200
    return jsonify(resp.to_dict()), status


# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'agents': bus.agents()})


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def create_app():
    return app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
