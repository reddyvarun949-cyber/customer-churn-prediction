"""Automated tests for the API. Run from the project root:

    python -m unittest discover -s backend -p "test_*.py" -v

They use Flask's built-in test client, so no server needs to be running.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app, THRESHOLD  # noqa: E402

# A risky customer: new, month-to-month, fibre without tech support, several complaints
RISKY = {
    "tenure_months": 2, "contract_type": "Month-to-month", "monthly_charges": 1050, "total_charges": 2100,
    "internet_service": "Fiber optic", "tech_support": "No", "payment_method": "Electronic check",
    "paperless_billing": "Yes", "num_complaints_6m": 3, "avg_data_gb": 90, "senior_citizen": 0, "city_tier": "Tier 1",
}
# A safe customer: long two-year contract, no complaints
LOYAL = {
    "tenure_months": 60, "contract_type": "Two year", "monthly_charges": 700, "total_charges": 42000,
    "internet_service": "DSL", "tech_support": "Yes", "payment_method": "Bank transfer (automatic)",
    "paperless_billing": "No", "num_complaints_6m": 0, "avg_data_gb": 45, "senior_citizen": 0, "city_tier": "Tier 2",
}


class ChurnApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    # ---- basic endpoints ----
    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_options(self):
        data = self.client.get("/api/options").get_json()
        self.assertIn("Month-to-month", data["categories"]["contract_type"])
        self.assertIn("tenure_months", data["numeric_ranges"])
        self.assertAlmostEqual(data["threshold"], THRESHOLD)
        self.assertIn("offer_cost_rupees", data["business_assumptions"])

    # ---- single prediction ----
    def test_valid_prediction(self):
        r = self.client.post("/api/predict", json=RISKY)
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(0 <= body["churn_probability"] <= 1)
        self.assertEqual(body["will_churn"], body["churn_probability"] >= THRESHOLD)
        self.assertIn(body["decision"], ["Contact", "Do not contact"])
        self.assertTrue(body["recommended_action"])
        print("\n  risky customer:", body)

    def test_risky_scores_higher_than_loyal(self):
        risky = self.client.post("/api/predict", json=RISKY).get_json()
        loyal = self.client.post("/api/predict", json=LOYAL).get_json()
        print("\n  loyal customer:", loyal)
        self.assertGreater(risky["churn_probability"], loyal["churn_probability"])
        self.assertTrue(risky["will_churn"])
        self.assertFalse(loyal["will_churn"])

    def test_numbers_sent_as_text_are_accepted(self):
        r = self.client.post("/api/predict", json={**RISKY, "tenure_months": "2", "monthly_charges": "1050"})
        self.assertEqual(r.status_code, 200)

    def test_blank_total_charges_for_new_customer(self):
        r = self.client.post("/api/predict", json={**RISKY, "tenure_months": 0, "total_charges": " "})
        self.assertEqual(r.status_code, 200)

    def test_blank_total_charges_for_old_customer_rejected(self):
        r = self.client.post("/api/predict", json={**RISKY, "total_charges": ""})
        self.assertEqual(r.status_code, 422)
        self.assertIn("total_charges", r.get_json()["fields"])

    def test_missing_optional_data_usage_is_allowed(self):
        payload = dict(RISKY)
        del payload["avg_data_gb"]
        self.assertEqual(self.client.post("/api/predict", json=payload).status_code, 200)

    def test_missing_required_field(self):
        payload = dict(RISKY)
        del payload["tenure_months"]
        r = self.client.post("/api/predict", json=payload)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["tenure_months"], "This field is required.")

    def test_not_a_number(self):
        r = self.client.post("/api/predict", json={**RISKY, "monthly_charges": "a lot"})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["fields"]["monthly_charges"], "Must be a number.")

    def test_out_of_range(self):
        r = self.client.post("/api/predict", json={**RISKY, "tenure_months": 500})
        self.assertEqual(r.status_code, 422)
        self.assertIn("between", r.get_json()["fields"]["tenure_months"])

    def test_whole_number_rule(self):
        r = self.client.post("/api/predict", json={**RISKY, "num_complaints_6m": 1.5})
        self.assertEqual(r.status_code, 422)
        self.assertIn("whole number", r.get_json()["fields"]["num_complaints_6m"])

    def test_unknown_category(self):
        r = self.client.post("/api/predict", json={**RISKY, "payment_method": "Cash at store"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("Unknown value", r.get_json()["fields"]["payment_method"])

    def test_inconsistent_tech_support(self):
        r = self.client.post("/api/predict", json={**RISKY, "internet_service": "No internet"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("tech_support", r.get_json()["fields"])

    def test_body_not_json(self):
        r = self.client.post("/api/predict", data="hello", content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_body_json_but_not_object(self):
        r = self.client.post("/api/predict", json=[1, 2, 3])
        self.assertEqual(r.status_code, 422)

    # ---- batch prediction ----
    def test_batch_with_mixed_rows(self):
        bad = {**LOYAL, "customer_id": "C-3", "contract_type": "Weekly"}
        r = self.client.post("/api/predict-batch", json={"customers": [
            {**RISKY, "customer_id": "C-1"}, {**LOYAL, "customer_id": "C-2"}, bad, "not an object"]})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["summary"], {"total": 4, "valid": 2, "invalid": 2, "to_contact": 1})
        results = body["results"]
        self.assertTrue(results[0]["ok"])
        self.assertEqual(results[0]["customer_id"], "C-1")
        self.assertFalse(results[2]["ok"])
        self.assertIn("contract_type", results[2]["fields"])
        self.assertEqual(results[3]["row"], 4)
        self.assertIn("body", results[3]["fields"])

    def test_batch_matches_single_prediction(self):
        single = self.client.post("/api/predict", json=RISKY).get_json()
        batch = self.client.post("/api/predict-batch", json=[RISKY]).get_json()   # a plain list also works
        self.assertAlmostEqual(single["churn_probability"], batch["results"][0]["churn_probability"])

    def test_batch_limit(self):
        r = self.client.post("/api/predict-batch", json={"customers": [RISKY] * 101})
        self.assertEqual(r.status_code, 422)
        self.assertIn("limit", r.get_json()["error"])

    def test_batch_empty_and_wrong_shape(self):
        self.assertEqual(self.client.post("/api/predict-batch", json={"customers": []}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict-batch", json={"rows": RISKY}).status_code, 422)

    def test_batch_body_not_json(self):
        r = self.client.post("/api/predict-batch", data="a,b,c", content_type="text/csv")
        self.assertEqual(r.status_code, 400)

    # ---- CORS ----
    def test_cors_headers_present(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        self.assertEqual(self.client.options("/api/predict").status_code, 204)
        self.assertEqual(self.client.options("/api/predict-batch").status_code, 204)


if __name__ == "__main__":
    unittest.main()
