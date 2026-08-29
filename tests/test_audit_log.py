import unittest
import os
import tempfile
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from audit_log import AuditLog, DuplicateAuditIdError, AuditLogError
from pipeline_utils import MODEL_VERSION

class TestAuditLog(unittest.TestCase):
    
    def setUp(self):
        # Use a temporary file for the database so tests do not touch logs/audit_log.db
        self.fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.fd) # Close file descriptor so sqlite3 can use it
        
        self.logger = AuditLog(db_path=self.temp_db_path)
        
        self.valid_params = {
            "order_id": "order_test_123",
            "risk_score": 0.72,
            "decision": "HOLD",
            "reason_codes": ["ELEVATED_RISK_PAYMENT_METHOD"],
            "model_version": MODEL_VERSION,
            "policy_version": "1.1"
        }

    def tearDown(self):
        try:
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except PermissionError:
            pass

    # 1. Initialization
    def test_database_initialization(self):
        # Verification that the table exists is implicitly done if append succeeds, 
        # but we can explicitly check that the file exists and has size.
        self.assertTrue(os.path.exists(self.temp_db_path))
        self.assertGreater(os.path.getsize(self.temp_db_path), 0)
        
    def test_timestamp_index_exists(self):
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.temp_db_path)) as conn, conn:
            cursor = conn.execute("PRAGMA index_list('decision_audit_log')")
            indices = [row[1] for row in cursor.fetchall()]
            self.assertIn('idx_audit_log_timestamp', indices)

    # 2 & 3. Append and Retrieve exact record
    def test_append_and_retrieve_valid_decision(self):
        audit_id = self.logger.append_decision(**self.valid_params)
        
        # Verify format and uniqueness
        self.assertTrue(audit_id.startswith("AUD-"))
        self.assertEqual(len(audit_id), 16)
        self.assertTrue(all(c in "0123456789ABCDEF" for c in audit_id[4:]))
        
        audit_id2 = self.logger.append_decision(**self.valid_params)
        self.assertNotEqual(audit_id, audit_id2)
        
        record = self.logger.get_audit_record(audit_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["audit_id"], audit_id)
        self.assertEqual(record["order_id"], self.valid_params["order_id"])
        self.assertAlmostEqual(record["risk_score"], self.valid_params["risk_score"])
        self.assertEqual(record["decision"], self.valid_params["decision"])
        self.assertEqual(record["reason_codes"], self.valid_params["reason_codes"])
        self.assertEqual(record["model_version"], self.valid_params["model_version"])
        self.assertEqual(record["policy_version"], self.valid_params["policy_version"])
        
        # Verify timestamp format (ISO 8601 UTC)
        self.assertIn("+00:00", record["timestamp"])

    # 4. Invalid risk score rejected
    def test_invalid_risk_score(self):
        invalid_scores = [-0.1, 1.1, float('nan'), float('inf'), "0.5", None]
        for score in invalid_scores:
            params = self.valid_params.copy()
            params["risk_score"] = score
            with self.assertRaises(ValueError):
                self.logger.append_decision(**params)

    # 5. Invalid decision rejected
    def test_invalid_decision(self):
        invalid_decisions = ["DECLINE", "approve", "REVIEWED", "", None, 1]
        for dec in invalid_decisions:
            params = self.valid_params.copy()
            params["decision"] = dec
            with self.assertRaises(ValueError):
                self.logger.append_decision(**params)

    # 6. Missing required fields rejected
    def test_missing_fields(self):
        for key in ["order_id", "model_version", "policy_version"]:
            params = self.valid_params.copy()
            params[key] = "" # empty string
            with self.assertRaises(ValueError):
                self.logger.append_decision(**params)
            
            params[key] = None
            with self.assertRaises(ValueError):
                self.logger.append_decision(**params)

    # 7. Malformed reason codes rejected
    def test_malformed_reason_codes(self):
        invalid_codes = ["just_a_string", None, [1, 2], {"code": "A"}]
        for codes in invalid_codes:
            params = self.valid_params.copy()
            params["reason_codes"] = codes
            with self.assertRaises(ValueError):
                self.logger.append_decision(**params)

    # 8. Append-only (Existing audit records cannot be overwritten)
    def test_append_only_duplicate_id(self):
        audit_id = self.logger.append_decision(**self.valid_params)
        
        # Attempt to insert same audit_id
        params = self.valid_params.copy()
        params["audit_id"] = audit_id
        params["decision"] = "APPROVE"
        
        with self.assertRaises(DuplicateAuditIdError):
            self.logger.append_decision(**params)
            
        # 10. Existing record remains unchanged after failure
        record = self.logger.get_audit_record(audit_id)
        self.assertEqual(record["decision"], "HOLD")

    # 9. Appending a second event creates a second record
    def test_multiple_appends(self):
        id1 = self.logger.append_decision(**self.valid_params)
        id2 = self.logger.append_decision(**self.valid_params)
        
        self.assertNotEqual(id1, id2)
        
        records = self.logger.list_recent_decisions(limit=10)
        self.assertEqual(len(records), 2)

    # 12. list_recent_decisions deterministic order
    def test_list_recent_decisions_ordering(self):
        # Insert 3 records
        id1 = self.logger.append_decision(**self.valid_params)
        id2 = self.logger.append_decision(**self.valid_params)
        id3 = self.logger.append_decision(**self.valid_params)
        
        records = self.logger.list_recent_decisions(limit=2)
        
        self.assertEqual(len(records), 2)
        # Should be most recent first (id3, then id2)
        self.assertEqual(records[0]["audit_id"], id3)
        self.assertEqual(records[1]["audit_id"], id2)

    # 13. Database/write errors surfaced
    def test_database_write_error_surfaced(self):
        # Make the database file strictly read-only to trigger an error
        import stat
        os.chmod(self.temp_db_path, stat.S_IREAD)
        
        with self.assertRaises(AuditLogError):
            self.logger.append_decision(**self.valid_params)
            
        # Restore permissions for teardown
        os.chmod(self.temp_db_path, stat.S_IWRITE | stat.S_IREAD)

if __name__ == '__main__':
    unittest.main()
