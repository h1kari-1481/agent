"""
SystemAgent – platform-level management agent.

Responsibilities:
  - User management (create, update, delete)
  - Role management
  - System monitoring (stub)
  - Route messages to the appropriate domain agent
"""
from __future__ import annotations
from datetime import datetime
from .base_agent import BaseAgent, Message


class SystemAgent(BaseAgent):
    """Platform administration and message-routing agent."""

    def _register_handlers(self):
        self.register('create_user', self._handle_create_user)
        self.register('delete_user', self._handle_delete_user)
        self.register('update_user', self._handle_update_user)
        self.register('list_users', self._handle_list_users)
        self.register('create_role', self._handle_create_role)
        self.register('list_roles', self._handle_list_roles)
        self.register('login', self._handle_login)
        self.register('system_status', self._handle_system_status)
        self.register('list_agent_messages', self._handle_list_messages)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _handle_login(self, msg: Message) -> Message:
        p = msg.payload
        row = self.db.execute(
            """SELECT u.id, u.username, u.region_code, r.name AS role_name
               FROM users u JOIN roles r ON r.id=u.role_id
               WHERE u.username=? AND u.password=?""",
            (p.get('username'), p.get('password')),
        ).fetchone()
        if row is None:
            return self._reply(msg, 'login_failed', {'error': '用户名或密码错误'})
        return self._reply(msg, 'login_success', {
            'user_id': row['id'],
            'username': row['username'],
            'role': row['role_name'],
            'region_code': row['region_code'],
        })

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def _handle_create_user(self, msg: Message) -> Message:
        p = msg.payload
        now = datetime.now().isoformat()
        role_row = self.db.execute(
            "SELECT id FROM roles WHERE name=?", (p.get('role_name', 'enterprise'),)
        ).fetchone()
        if role_row is None:
            return self._reply(msg, 'error', {'error': 'Role not found'})
        try:
            cur = self.db.execute(
                "INSERT INTO users(username, password, role_id, region_code, created_at) VALUES (?,?,?,?,?)",
                (p['username'], p['password'], role_row['id'], p.get('region_code', ''), now),
            )
            self.db.commit()
        except Exception as e:
            return self._reply(msg, 'error', {'error': str(e)})
        self._persist_message(msg)
        return self._reply(msg, 'user_created', {'user_id': cur.lastrowid})

    def _handle_delete_user(self, msg: Message) -> Message:
        p = msg.payload
        uid = p.get('user_id')
        # Check if user has submissions
        count = self.db.execute(
            "SELECT COUNT(*) AS c FROM submissions WHERE submitted_by=?", (uid,)
        ).fetchone()['c']
        if count > 0:
            return self._reply(msg, 'error', {'error': '该用户有上报数据，不能删除'})
        self.db.execute("DELETE FROM users WHERE id=?", (uid,))
        self.db.commit()
        return self._reply(msg, 'user_deleted', {'user_id': uid})

    def _handle_update_user(self, msg: Message) -> Message:
        p = msg.payload
        uid = p.get('user_id')
        self.db.execute(
            "UPDATE users SET password=COALESCE(?,password), region_code=COALESCE(?,region_code) WHERE id=?",
            (p.get('password'), p.get('region_code'), uid),
        )
        self.db.commit()
        return self._reply(msg, 'user_updated', {'user_id': uid})

    def _handle_list_users(self, msg: Message) -> Message:
        rows = self.db.execute(
            """SELECT u.id, u.username, r.name AS role_name, u.region_code, u.created_at
               FROM users u JOIN roles r ON r.id=u.role_id ORDER BY u.id"""
        ).fetchall()
        return self._reply(msg, 'user_list', {'users': [dict(r) for r in rows]})

    # ------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------

    def _handle_create_role(self, msg: Message) -> Message:
        p = msg.payload
        try:
            cur = self.db.execute(
                "INSERT INTO roles(name, desc) VALUES (?,?)",
                (p['name'], p.get('desc', '')),
            )
            self.db.commit()
        except Exception as e:
            return self._reply(msg, 'error', {'error': str(e)})
        return self._reply(msg, 'role_created', {'role_id': cur.lastrowid})

    def _handle_list_roles(self, msg: Message) -> Message:
        rows = self.db.execute("SELECT id, name, desc FROM roles").fetchall()
        return self._reply(msg, 'role_list', {'roles': [dict(r) for r in rows]})

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def _handle_system_status(self, msg: Message) -> Message:
        import platform, psutil
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            status = {
                'cpu_percent': cpu,
                'memory_total_mb': round(mem.total / 1024 / 1024),
                'memory_used_mb': round(mem.used / 1024 / 1024),
                'disk_total_gb': round(disk.total / 1024 / 1024 / 1024, 1),
                'disk_used_gb': round(disk.used / 1024 / 1024 / 1024, 1),
                'platform': platform.system(),
                'python': platform.python_version(),
            }
        except ImportError:
            status = {'note': 'psutil not installed – monitoring unavailable'}
        return self._reply(msg, 'system_status', {'status': status})

    def _handle_list_messages(self, msg: Message) -> Message:
        p = msg.payload
        limit = p.get('limit', 50)
        rows = self.db.execute(
            "SELECT * FROM agent_messages ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return self._reply(msg, 'message_list', {'messages': [dict(r) for r in rows]})
