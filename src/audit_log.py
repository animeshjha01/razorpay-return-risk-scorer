import sqlite3
import json
import uuid
import math
from datetime import datetime, timezone
import os
from contextlib import closing

class AuditLogError(Exception):
    """Base exception for audit log operations."""
    pass

class DuplicateAuditIdError(AuditLogError):
    """Raised when an attempt is made to insert an existing audit ID."""
    pass

class AuditLog:
    def __init__(self, db_path: str = "logs/audit_log.db"):
        self.db_path = db_path
        self._ensure_directories()
        self._initialize_db()

    def _ensure_directories(self):
        """Creates the parent directory for the database if it doesn't exist."""
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _initialize_db(self):
        """Creates the explicit audit table with schema and constraints."""
        schema = """
        CREATE TABLE IF NOT EXISTS decision_audit_log (
            audit_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            order_id TEXT NOT NULL,
            risk_score REAL NOT NULL,
            decision TEXT NOT NULL,
            reason_codes TEXT NOT NULL,
            model_version TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            CHECK (decision IN ('APPROVE', 'REVIEW', 'HOLD'))
        );
        CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON decision_audit_log (timestamp DESC);
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                conn.executescript(schema)
        except sqlite3.Error as e:
            raise AuditLogError(f"Failed to initialize database schema: {e}")

    def append_decision(
        self,
        order_id: str,
        risk_score: float,
        decision: str,
        reason_codes: list,
        model_version: str,
        policy_version: str,
        audit_id: str = None
    ) -> str:
        """
        Records a scoring decision in the append-only audit log.
        Returns the generated (or verified) audit_id.
        """
        # 1. Input Validation
        if not order_id or not isinstance(order_id, str):
            raise ValueError("order_id must be a non-empty string.")
            
        if not isinstance(risk_score, (float, int)):
            raise ValueError("risk_score must be a numeric value.")
        if math.isnan(risk_score) or math.isinf(risk_score) or not (0.0 <= risk_score <= 1.0):
            raise ValueError(f"risk_score must be a finite float between 0.0 and 1.0, got: {risk_score}")
            
        if decision not in ('APPROVE', 'REVIEW', 'HOLD'):
            raise ValueError(f"decision must be APPROVE, REVIEW, or HOLD, got: {decision}")
            
        if not isinstance(reason_codes, list) or not all(isinstance(c, str) for c in reason_codes):
            raise ValueError("reason_codes must be a list of strings.")
            
        if not model_version or not isinstance(model_version, str):
            raise ValueError("model_version must be a non-empty string.")
            
        if not policy_version or not isinstance(policy_version, str):
            raise ValueError("policy_version must be a non-empty string.")

        # 2. Generation of system fields
        final_audit_id = audit_id if audit_id else f"AUD-{uuid.uuid4().hex[:12].upper()}"
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        reason_codes_json = json.dumps(reason_codes)

        # 3. Insertion (Append-only)
        query = """
        INSERT INTO decision_audit_log (
            audit_id, timestamp, order_id, risk_score, decision, reason_codes, model_version, policy_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                conn.execute(query, (
                    final_audit_id,
                    timestamp_iso,
                    order_id,
                    float(risk_score),
                    decision,
                    reason_codes_json,
                    model_version,
                    policy_version
                ))
        except sqlite3.IntegrityError as e:
            if 'unique constraint failed' in str(e).lower() or 'primary key' in str(e).lower():
                raise DuplicateAuditIdError(f"Audit ID '{final_audit_id}' already exists.")
            raise AuditLogError(f"Integrity error during insert: {e}")
        except sqlite3.Error as e:
            raise AuditLogError(f"Database write failure: {e}")

        return final_audit_id

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Helper to safely deserialize a database row into a dict."""
        d = dict(row)
        try:
            d['reason_codes'] = json.loads(d['reason_codes'])
        except json.JSONDecodeError:
            d['reason_codes'] = []
        return d

    def get_audit_record(self, audit_id: str) -> dict:
        """
        Retrieves a single audit record by its audit_id.
        Returns None if not found.
        """
        query = "SELECT * FROM decision_audit_log WHERE audit_id = ?"
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (audit_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_dict(row)
                return None
        except sqlite3.Error as e:
            raise AuditLogError(f"Database read failure: {e}")

    def list_recent_decisions(self, limit: int = 10) -> list[dict]:
        """
        Retrieves the most recent audit decisions in deterministic descending order.
        """
        query = "SELECT * FROM decision_audit_log ORDER BY timestamp DESC LIMIT ?"
        try:
            with closing(sqlite3.connect(self.db_path)) as conn, conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (limit,))
                return [self._row_to_dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise AuditLogError(f"Database read failure: {e}")
