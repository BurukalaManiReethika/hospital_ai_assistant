import unittest

from rag import answer, get_retriever


class TestRetrieval(unittest.TestCase):
    def top_id(self, q):
        return get_retriever().search(q, k=1)[0]["id"]

    def test_faq_opd_timings(self):
        self.assertEqual(self.top_id("What are the OPD timings?"), "opd-timings")

    def test_faq_booking(self):
        self.assertEqual(self.top_id("How do I book an appointment?"), "book-appointment")

    def test_faq_insurance(self):
        self.assertEqual(self.top_id("Do you accept insurance?"), "insurance")

    def test_faq_consultation_fee(self):
        self.assertEqual(self.top_id("how much is a consultation"), "consultation-fee")

    def test_faq_pharmacy(self):
        self.assertEqual(self.top_id("pharmacy open at night?"), "pharmacy")

    def test_doctor_heart(self):
        self.assertIn("Anjali Rao", answer("Which doctor treats heart problems?")["answer"])

    def test_doctor_child(self):
        self.assertIn("Meena Iyer", answer("Is there a child specialist?")["answer"])

    def test_doctor_bone_singular(self):
        self.assertIn("Suresh Reddy", answer("bone doctor timings")["answer"])

    def test_doctor_availability_phrasing(self):
        self.assertIn("Anjali Rao", answer("When is the cardiologist available?")["answer"])

    def test_doctor_skin(self):
        self.assertIn("Imran Sheikh", answer("I need a skin doctor")["answer"])


class TestAnswers(unittest.TestCase):
    def test_answer_has_sources(self):
        r = answer("What are the visiting hours?")
        self.assertTrue(r["sources"])
        self.assertIn("4:00 PM", r["answer"])

    def test_out_of_scope_falls_back(self):
        self.assertEqual(answer("What is the capital of France?")["mode"], "fallback")

    def test_emergency(self):
        self.assertEqual(answer("my father has chest pain")["mode"], "emergency")

    def test_medical_advice_declined(self):
        self.assertEqual(answer("Which medicine should I take for fever?")["mode"], "safety")

    def test_empty_input(self):
        self.assertEqual(answer("   ")["mode"], "empty")


if __name__ == "__main__":
    unittest.main()
