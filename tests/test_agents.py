"""
Comprehensive test suite for all four agents.

Run with: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sqlite3
import pytest

from src.models.database import init_db
from src.agents.base_agent import Message
from src.agents.agent_bus import AgentBus
from src.agents.enterprise_agent import EnterpriseAgent
from src.agents.city_agent import CityAgent
from src.agents.province_agent import ProvinceAgent
from src.agents.system_agent import SystemAgent


def _make_conn():
    """Create a fresh in-memory SQLite connection with the full schema."""
    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(':memory:', check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
    CREATE TABLE roles (
        id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT    NOT NULL UNIQUE,
        desc TEXT
    );
    CREATE TABLE users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        username    TEXT    NOT NULL UNIQUE,
        password    TEXT    NOT NULL,
        role_id     INTEGER NOT NULL REFERENCES roles(id),
        region_code TEXT,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE enterprises (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        org_code        TEXT    NOT NULL UNIQUE,
        name            TEXT    NOT NULL,
        nature          TEXT,
        industry        TEXT,
        region_code     TEXT    NOT NULL,
        contact_person  TEXT,
        contact_phone   TEXT,
        email           TEXT,
        address         TEXT,
        main_business   TEXT,
        status          TEXT    NOT NULL DEFAULT 'pending',
        created_by      INTEGER REFERENCES users(id),
        created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE survey_periods (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        period_key TEXT    NOT NULL UNIQUE,
        start_date TEXT    NOT NULL,
        end_date   TEXT    NOT NULL,
        created_by INTEGER REFERENCES users(id)
    );
    CREATE TABLE submissions (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        enterprise_id           INTEGER NOT NULL REFERENCES enterprises(id),
        period_id               INTEGER NOT NULL REFERENCES survey_periods(id),
        baseline_employees      INTEGER NOT NULL,
        survey_employees        INTEGER NOT NULL,
        decrease_type           TEXT,
        main_reason             TEXT,
        main_reason_desc        TEXT,
        secondary_reason        TEXT,
        secondary_reason_desc   TEXT,
        third_reason            TEXT,
        third_reason_desc       TEXT,
        status                  TEXT NOT NULL DEFAULT 'draft',
        submitted_by            INTEGER REFERENCES users(id),
        submitted_at            TEXT,
        city_reviewed_by        INTEGER REFERENCES users(id),
        city_reviewed_at        TEXT,
        province_reviewed_by    INTEGER REFERENCES users(id),
        province_reviewed_at    TEXT,
        return_reason           TEXT,
        created_at              TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE notifications (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        title        TEXT    NOT NULL,
        content      TEXT    NOT NULL,
        published_by INTEGER NOT NULL REFERENCES users(id),
        published_at TEXT    NOT NULL DEFAULT (datetime('now')),
        is_deleted   INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE agent_messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        sender       TEXT    NOT NULL,
        receiver     TEXT    NOT NULL,
        msg_type     TEXT    NOT NULL,
        payload      TEXT    NOT NULL,
        status       TEXT    NOT NULL DEFAULT 'pending',
        created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
        processed_at TEXT
    );
    """)
    conn.executemany(
        "INSERT OR IGNORE INTO roles(name, desc) VALUES (?, ?)",
        [('province_admin', '省级管理员'), ('city_admin', '市级管理员'), ('enterprise', '企业用户')],
    )
    conn.execute(
        "INSERT INTO users(username, password, role_id, region_code) "
        "VALUES ('admin', 'admin123', (SELECT id FROM roles WHERE name='province_admin'), '53')"
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def bus():
    """Create an in-memory SQLite database and wire up all agents."""
    conn = _make_conn()
    b = AgentBus()
    b.register_agent(SystemAgent('system', conn))
    b.register_agent(EnterpriseAgent('enterprise', conn))
    b.register_agent(CityAgent('city', conn))
    b.register_agent(ProvinceAgent('province', conn))
    yield b
    conn.close()


def send(bus, agent, action, **kwargs):
    msg = Message('test', agent, action, kwargs)
    return bus.send(msg)


# ---------------------------------------------------------------------------
# SystemAgent tests
# ---------------------------------------------------------------------------

class TestSystemAgent:
    def test_login_success(self, bus):
        resp = send(bus, 'system', 'login', username='admin', password='admin123')
        assert resp.msg_type == 'login_success'
        assert resp.payload['role'] == 'province_admin'

    def test_login_failure(self, bus):
        resp = send(bus, 'system', 'login', username='admin', password='wrong')
        assert resp.msg_type == 'login_failed'

    def test_create_user(self, bus):
        resp = send(bus, 'system', 'create_user',
                    username='city_user', password='pass123',
                    role_name='city_admin', region_code='5301')
        assert resp.msg_type == 'user_created'
        assert 'user_id' in resp.payload

    def test_list_users(self, bus):
        resp = send(bus, 'system', 'list_users')
        assert resp.msg_type == 'user_list'
        assert any(u['username'] == 'admin' for u in resp.payload['users'])

    def test_list_roles(self, bus):
        resp = send(bus, 'system', 'list_roles')
        assert resp.msg_type == 'role_list'
        names = [r['name'] for r in resp.payload['roles']]
        assert 'province_admin' in names
        assert 'enterprise' in names


# ---------------------------------------------------------------------------
# EnterpriseAgent tests
# ---------------------------------------------------------------------------

class TestEnterpriseAgent:
    @pytest.fixture(autouse=True)
    def _create_enterprise_user(self, bus):
        resp = send(bus, 'system', 'create_user',
                    username='ent_user1', password='pass123',
                    role_name='enterprise', region_code='5301')
        self.user_id = resp.payload.get('user_id')

    def test_register_enterprise(self, bus):
        resp = send(bus, 'enterprise', 'register_enterprise',
                    user_id=self.user_id,
                    org_code='ENT001',
                    name='测试科技有限公司',
                    nature='私营',
                    industry='信息技术',
                    region_code='5301',
                    contact_person='张三',
                    contact_phone='0871-12345678',
                    email='test@test.com',
                    address='昆明市盘龙区')
        assert resp.msg_type == 'enterprise_registered'
        assert resp.payload['status'] == 'pending'

    def test_register_invalid_org_code(self, bus):
        resp = send(bus, 'enterprise', 'register_enterprise',
                    user_id=self.user_id,
                    org_code='TOOLONGCODE99',  # > 9 chars
                    name='Another Co',
                    region_code='5301')
        assert resp.msg_type == 'error'

    def test_browse_notifications(self, bus):
        resp = send(bus, 'enterprise', 'browse_notifications')
        assert resp.msg_type == 'notification_list'

    def test_query_submissions_empty(self, bus):
        resp = send(bus, 'enterprise', 'query_submissions', enterprise_id=1)
        assert resp.msg_type == 'submission_list'


# ---------------------------------------------------------------------------
# ProvinceAgent tests
# ---------------------------------------------------------------------------

class TestProvinceAgent:
    @pytest.fixture(autouse=True)
    def _setup(self, bus):
        # Create survey period
        resp = send(bus, 'province', 'create_survey_period',
                    user_id=1,
                    period_key='2026-01',
                    start_date='2026-01-01',
                    end_date='2026-01-31')
        self.period_id = resp.payload.get('period_id')

        # Approve the enterprise created in enterprise tests
        send(bus, 'province', 'approve_enterprise', enterprise_id=1)

    def test_create_survey_period(self, bus):
        resp = send(bus, 'province', 'create_survey_period',
                    user_id=1,
                    period_key='2026-02',
                    start_date='2026-02-01',
                    end_date='2026-02-28')
        assert resp.msg_type == 'period_created'

    def test_approve_enterprise(self, bus):
        resp = send(bus, 'province', 'approve_enterprise', enterprise_id=1)
        assert resp.msg_type == 'enterprise_approved'

    def test_publish_notification(self, bus):
        resp = send(bus, 'province', 'publish_notification',
                    user_id=1,
                    title='测试通知',
                    content='这是一条测试通知内容')
        assert resp.msg_type == 'notification_published'

    def test_publish_notification_title_too_long(self, bus):
        resp = send(bus, 'province', 'publish_notification',
                    user_id=1,
                    title='a' * 51,
                    content='content')
        assert resp.msg_type == 'error'

    def test_aggregate_report_empty(self, bus):
        resp = send(bus, 'province', 'aggregate_report', period_key='2026-01')
        assert resp.msg_type == 'aggregate_report'
        assert 'report' in resp.payload


# ---------------------------------------------------------------------------
# End-to-end data submission workflow
# ---------------------------------------------------------------------------

class TestSubmissionWorkflow:
    def test_full_workflow(self, bus):
        # 1. Enterprise submits data
        resp = send(bus, 'enterprise', 'submit_data',
                    user_id=self.user_id if hasattr(self, 'user_id') else 1,
                    enterprise_id=1,
                    period_id=1,
                    baseline_employees=100,
                    survey_employees=90,
                    decrease_type='经济性裁员',
                    main_reason='订单不足',
                    main_reason_desc='受市场影响订单减少')
        assert resp.msg_type == 'data_submitted'
        sub_id = resp.payload['submission_id']

        # 2. City agent approves
        resp2 = send(bus, 'city', 'approve_submission',
                     submission_id=sub_id, user_id=1)
        assert resp2.msg_type == 'submission_approved'
        assert resp2.payload['level'] == 'city'

        # 3. Province agent approves
        resp3 = send(bus, 'province', 'approve_submission',
                     submission_id=sub_id, user_id=1)
        assert resp3.msg_type == 'submission_approved'
        assert resp3.payload['level'] == 'province'

    def test_submit_missing_reason(self, bus):
        """Submitting with survey < baseline but no decrease reason should fail."""
        resp = send(bus, 'enterprise', 'submit_data',
                    user_id=1, enterprise_id=1, period_id=1,
                    baseline_employees=100,
                    survey_employees=50)
        assert resp.msg_type == 'error'

    def test_return_submission(self, bus):
        """City agent returns a submission for correction."""
        # Submit fresh
        s = send(bus, 'enterprise', 'submit_data',
                 user_id=1, enterprise_id=1, period_id=1,
                 baseline_employees=80, survey_employees=80)
        sub_id = s.payload['submission_id']
        r = send(bus, 'city', 'return_submission',
                 submission_id=sub_id, user_id=1, return_reason='数据有误')
        assert r.msg_type == 'submission_returned'
        assert r.payload['return_reason'] == '数据有误'
