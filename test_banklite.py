import pytest
from unittest.mock import MagicMock, patch

from banklite import (
    Transaction,
    FraudCheckResult,
    PaymentGateway,
    FraudDetector,
    EmailClient,
    AuditLog,
    TransactionRepository,
    PaymentProcessor,
    FraudAwareProcessor,
    StatementBuilder,
    FeeCalculator,
    CheckoutService,
)


# ── Task 1: PaymentProcessor ─────────────────────────────────
class TestPaymentProcessor:
    def setup_method(self):
        self.gateway = MagicMock(spec=PaymentGateway)
        self.audit = MagicMock(spec=AuditLog)
        self.processor = PaymentProcessor(self.gateway, self.audit)

    def test_process_valid_transaction_success_returns_success(self):
        tx = Transaction(tx_id="tx100", user_id=1, amount=50.00)
        self.gateway.charge.return_value = True

        result = self.processor.process(tx)

        assert result == "success"
        self.gateway.charge.assert_called_once_with(tx)

    def test_process_valid_transaction_declined_returns_declined(self):
        tx = Transaction(tx_id="tx101", user_id=1, amount=50.00)
        self.gateway.charge.return_value = False

        result = self.processor.process(tx)

        assert result == "declined"
        self.gateway.charge.assert_called_once_with(tx)

    def test_process_zero_amount_raises_value_error(self):
        tx = Transaction(tx_id="tx102", user_id=1, amount=0.0)

        with pytest.raises(ValueError):
            self.processor.process(tx)

        self.gateway.charge.assert_not_called()

    def test_process_negative_amount_raises_value_error(self):
        tx = Transaction(tx_id="tx103", user_id=1, amount=-5.0)

        with pytest.raises(ValueError):
            self.processor.process(tx)

        self.gateway.charge.assert_not_called()

    def test_process_amount_exceeds_limit_raises_value_error(self):
        tx = Transaction(tx_id="tx104", user_id=1, amount=10001.00)

        with pytest.raises(ValueError):
            self.processor.process(tx)

        self.gateway.charge.assert_not_called()

    def test_process_success_records_charged_audit_event(self):
        tx = Transaction(tx_id="tx105", user_id=1, amount=20.00)
        self.gateway.charge.return_value = True

        result = self.processor.process(tx)

        assert result == "success"
        self.audit.record.assert_called_once_with(
            "CHARGED",
            "tx105",
            {"amount": 20.00},
        )

    def test_process_declined_records_declined_audit_event(self):
        tx = Transaction(tx_id="tx106", user_id=1, amount=20.00)
        self.gateway.charge.return_value = False

        result = self.processor.process(tx)

        assert result == "declined"
        self.audit.record.assert_called_once_with(
            "DECLINED",
            "tx106",
            {"amount": 20.00},
        )

    def test_process_invalid_input_does_not_call_audit(self):
        tx = Transaction(tx_id="tx107", user_id=1, amount=0.0)

        with pytest.raises(ValueError):
            self.processor.process(tx)

        self.audit.record.assert_not_called()


# ── Task 2: FraudAwareProcessor ──────────────────────────────
class TestFraudAwareProcessor:
    def setup_method(self):
        self.gateway = MagicMock(spec=PaymentGateway)
        self.detector = MagicMock(spec=FraudDetector)
        self.mailer = MagicMock(spec=EmailClient)
        self.audit = MagicMock(spec=AuditLog)
        self.processor = FraudAwareProcessor(
            self.gateway,
            self.detector,
            self.mailer,
            self.audit,
        )

    def test_process_high_risk_score_blocks_transaction(self):
        tx = Transaction(tx_id="tx200", user_id=2, amount=100.00)
        self.detector.check.return_value = FraudCheckResult(
            approved=False,
            risk_score=0.90,
            reason="high risk",
        )

        result = self.processor.process(tx)

        assert result == "blocked"
        self.gateway.charge.assert_not_called()
        self.mailer.send_fraud_alert.assert_called_once_with(2, "tx200")
        self.audit.record.assert_called_once_with(
            "BLOCKED",
            "tx200",
            {"risk": 0.90},
        )

    def test_process_exact_threshold_blocks_transaction(self):
        tx = Transaction(tx_id="tx201", user_id=2, amount=100.00)
        self.detector.check.return_value = FraudCheckResult(
            approved=False,
            risk_score=0.75,
            reason="threshold hit",
        )

        result = self.processor.process(tx)

        assert result == "blocked"
        self.gateway.charge.assert_not_called()
        self.mailer.send_fraud_alert.assert_called_once_with(2, "tx201")

    def test_process_low_risk_successful_charge_sends_receipt(self):
        tx = Transaction(tx_id="tx202", user_id=2, amount=80.00)
        self.detector.check.return_value = FraudCheckResult(
            approved=True,
            risk_score=0.20,
            reason="safe",
        )
        self.gateway.charge.return_value = True

        result = self.processor.process(tx)

        assert result == "success"
        self.gateway.charge.assert_called_once_with(tx)
        self.mailer.send_receipt.assert_called_once_with(2, "tx202", 80.00)
        self.mailer.send_fraud_alert.assert_not_called()
        self.audit.record.assert_called_once_with(
            "CHARGED",
            "tx202",
            {"amount": 80.00},
        )

    def test_process_low_risk_declined_charge_records_declined(self):
        tx = Transaction(tx_id="tx203", user_id=2, amount=80.00)
        self.detector.check.return_value = FraudCheckResult(
            approved=True,
            risk_score=0.10,
            reason="safe",
        )
        self.gateway.charge.return_value = False

        result = self.processor.process(tx)

        assert result == "declined"
        self.gateway.charge.assert_called_once_with(tx)
        self.mailer.send_receipt.assert_not_called()
        self.mailer.send_fraud_alert.assert_not_called()
        self.audit.record.assert_called_once_with(
            "DECLINED",
            "tx203",
            {"amount": 80.00},
        )

    def test_process_detector_connection_error_propagates_exception(self):
        tx = Transaction(tx_id="tx204", user_id=2, amount=60.00)
        self.detector.check.side_effect = ConnectionError("fraud API down")

        with pytest.raises(ConnectionError):
            self.processor.process(tx)

        self.gateway.charge.assert_not_called()
        self.mailer.send_receipt.assert_not_called()
        self.mailer.send_fraud_alert.assert_not_called()
        self.audit.record.assert_not_called()

    def test_process_high_risk_fraud_alert_uses_correct_args(self):
        tx = Transaction(tx_id="tx205", user_id=42, amount=90.00)
        self.detector.check.return_value = FraudCheckResult(
            approved=False,
            risk_score=0.80,
            reason="suspicious",
        )

        self.processor.process(tx)

        self.mailer.send_fraud_alert.assert_called_once_with(42, "tx205")

    def test_process_success_receipt_uses_correct_args(self):
        tx = Transaction(tx_id="tx206", user_id=77, amount=125.50)
        self.detector.check.return_value = FraudCheckResult(
            approved=True,
            risk_score=0.05,
            reason="safe",
        )
        self.gateway.charge.return_value = True

        self.processor.process(tx)

        self.mailer.send_receipt.assert_called_once_with(77, "tx206", 125.50)


# ── Task 3: StatementBuilder ─────────────────────────────────
class TestStatementBuilder:
    def setup_method(self):
        self.repo = MagicMock(spec=TransactionRepository)
        self.builder = StatementBuilder(self.repo)

    def test_build_no_transactions_returns_zero_count_and_total(self):
        self.repo.find_by_user.return_value = []

        result = self.builder.build(1)

        assert result["transactions"] == []
        assert result["count"] == 0
        assert result["total_charged"] == 0

    def test_build_only_success_transactions_sums_total_correctly(self):
        txs = [
            Transaction(tx_id="s1", user_id=1, amount=10.00, status="success"),
            Transaction(tx_id="s2", user_id=1, amount=15.25, status="success"),
        ]
        self.repo.find_by_user.return_value = txs

        result = self.builder.build(1)

        assert result["count"] == 2
        assert result["total_charged"] == 25.25

    def test_build_mixed_statuses_counts_only_success_amounts(self):
        txs = [
            Transaction(tx_id="m1", user_id=1, amount=10.00, status="success"),
            Transaction(tx_id="m2", user_id=1, amount=20.00, status="declined"),
            Transaction(tx_id="m3", user_id=1, amount=5.00, status="pending"),
            Transaction(tx_id="m4", user_id=1, amount=7.50, status="success"),
        ]
        self.repo.find_by_user.return_value = txs

        result = self.builder.build(1)

        assert result["count"] == 4
        assert result["total_charged"] == 17.50

    def test_build_rounds_total_to_two_decimal_places(self):
        txs = [
            Transaction(tx_id="r1", user_id=1, amount=10.555, status="success"),
            Transaction(tx_id="r2", user_id=1, amount=0.005, status="success"),
        ]
        self.repo.find_by_user.return_value = txs

        result = self.builder.build(1)

        assert result["total_charged"] == 10.56

    def test_build_returns_original_transactions_list_as_is(self):
        txs = [
            Transaction(tx_id="a1", user_id=1, amount=10.00, status="success"),
            Transaction(tx_id="a2", user_id=1, amount=12.00, status="declined"),
        ]
        self.repo.find_by_user.return_value = txs

        result = self.builder.build(1)

        assert result["transactions"] is txs


# ── Task 4: CheckoutService with spy ─────────────────────────
class TestCheckoutServiceWithSpy:
    def setup_method(self):
        self.real_fee_calc = FeeCalculator()
        self.fee_calc_spy = MagicMock(wraps=self.real_fee_calc)
        self.gateway = MagicMock(spec=PaymentGateway)
        self.service = CheckoutService(self.fee_calc_spy, self.gateway)

    def test_checkout_usd_transaction_computes_correct_fee(self):
        tx = Transaction(tx_id="tx300", user_id=3, amount=100.00, currency="USD")
        self.gateway.charge.return_value = True

        receipt = self.service.checkout(tx)

        assert receipt["fee"] == 3.20

    def test_checkout_international_transaction_computes_correct_fee(self):
        tx = Transaction(tx_id="tx301", user_id=3, amount=200.00, currency="EUR")
        self.gateway.charge.return_value = True

        receipt = self.service.checkout(tx)

        assert receipt["fee"] == 9.10

    def test_checkout_calls_processing_fee_with_correct_args(self):
        tx = Transaction(tx_id="tx302", user_id=3, amount=100.00, currency="USD")
        self.gateway.charge.return_value = True

        self.service.checkout(tx)

        self.fee_calc_spy.processing_fee.assert_called_once_with(100.00, "USD")

    def test_checkout_calls_net_amount_with_correct_args(self):
        tx = Transaction(tx_id="tx303", user_id=3, amount=200.00, currency="EUR")
        self.gateway.charge.return_value = True

        self.service.checkout(tx)

        self.fee_calc_spy.net_amount.assert_called_once_with(200.00, "EUR")

    def test_checkout_calls_each_fee_method_exactly_once(self):
        tx = Transaction(tx_id="tx304", user_id=3, amount=75.00, currency="USD")
        self.gateway.charge.return_value = True

        self.service.checkout(tx)

        assert self.fee_calc_spy.processing_fee.call_count == 1
        assert self.fee_calc_spy.net_amount.call_count == 1

    def test_checkout_real_fee_value_flows_into_receipt(self):
        tx = Transaction(tx_id="tx305", user_id=3, amount=50.00, currency="USD")
        self.gateway.charge.return_value = True

        receipt = self.service.checkout(tx)
        expected_fee = round(50.00 * 0.029 + 0.30, 2)

        assert receipt["fee"] == expected_fee

    def test_checkout_partial_spy_on_net_amount_observes_real_result(self):
        real_fee_calc = FeeCalculator()
        gateway = MagicMock(spec=PaymentGateway)
        gateway.charge.return_value = True
        service = CheckoutService(real_fee_calc, gateway)
        tx = Transaction(tx_id="tx306", user_id=3, amount=100.00, currency="USD")

        with patch.object(
            real_fee_calc,
            "net_amount",
            wraps=real_fee_calc.net_amount,
        ) as net_amount_spy:
            receipt = service.checkout(tx)

            net_amount_spy.assert_called_once_with(100.00, "USD")
            assert receipt["net"] == 96.80

    def test_checkout_with_plain_mock_cannot_verify_real_formula(self):
        # This test can verify delegation and returned values,
        # but it cannot prove the fee formula itself is correct,
        # because the values are hardcoded instead of coming from real logic.
        fake_fee_calc = MagicMock()
        fake_fee_calc.processing_fee.return_value = 999.99
        fake_fee_calc.net_amount.return_value = 0.01

        gateway = MagicMock(spec=PaymentGateway)
        gateway.charge.return_value = True

        service = CheckoutService(fake_fee_calc, gateway)
        tx = Transaction(tx_id="tx307", user_id=3, amount=100.00, currency="USD")

        receipt = service.checkout(tx)

        fake_fee_calc.processing_fee.assert_called_once_with(100.00, "USD")
        fake_fee_calc.net_amount.assert_called_once_with(100.00, "USD")
        assert receipt["fee"] == 999.99
        assert receipt["net"] == 0.01


# ── Task 5: Stretch / design notes ───────────────────────────

# Question A:
# StatementBuilder should mainly be tested by checking returned state.
# You could turn it into a mock-heavy test by asserting that repo.find_by_user()
# was called and focusing only on interactions, but that would miss the real point:
# StatementBuilder is a transformation class, so the important thing is whether
# the returned dict is correct, not whether a collaborator method was merely called.

def test_statement_builder_state_based_example():
    repo = MagicMock(spec=TransactionRepository)
    repo.find_by_user.return_value = [
        Transaction(tx_id="qA1", user_id=1, amount=12.00, status="success"),
        Transaction(tx_id="qA2", user_id=1, amount=5.00, status="declined"),
    ]
    builder = StatementBuilder(repo)

    result = builder.build(1)

    assert result["total_charged"] == 12.00
    assert result["count"] == 2


# Question B:
# spec= prevents you from accidentally using methods that do not exist on the real class.

def test_payment_gateway_spec_prevents_invalid_methods():
    gateway = MagicMock(spec=PaymentGateway)

    with pytest.raises(AttributeError):
        gateway.refund("tx999")


# Question C:
# @patch is useful when production code creates its own dependencies internally
# and you cannot inject them through the constructor.

@patch("banklite.FraudDetector")
def test_patch_example_for_fraud_detector_class(mock_detector_class):
    detector_instance = mock_detector_class.return_value
    detector_instance.check.return_value = FraudCheckResult(
        approved=False,
        risk_score=0.99,
        reason="patched detector",
    )

    tx = Transaction(tx_id="tx400", user_id=4, amount=70.00)

    result = detector_instance.check(tx)

    assert result.risk_score == 0.99


# Question D:
# A spy would usually NOT be a good choice for FraudDetector in FraudAwareProcessor
# because FraudDetector represents an external dependency, not a pure piece of logic
# we safely own and want to observe. Spies are best when real code is deterministic,
# side-effect-free, and safe to run in tests, like FeeCalculator. A real fraud detector
# may call an API, depend on network state, return changing values, or raise external
# errors unpredictably. In that case, a mock or stub is the better choice because it
# gives full control and keeps the test isolated.
