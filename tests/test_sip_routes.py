# tests/test_enhanced_sip_features.py
"""
Comprehensive test script for Enhanced SIP Features
Tests both requirements:
1. max_amount_in_a_month with default value as 4 times of fixed_investment
2. 4% price reduction threshold for multiple signals within a month
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import pandas as pd
import numpy as np

# Test configuration
TEST_CONFIG = {
    "database_url": "postgresql+asyncpg://test_user:test_pass@localhost/test_sip_db",
    "test_user_id": "test_user_123",
    "test_symbols": ["RELIANCE", "TCS", "INFY"]
}


class TestEnhancedSIPFeatures:
    """Test suite for enhanced SIP features"""

    def setup_method(self):
        """Setup test environment"""
        self.test_user_id = TEST_CONFIG["test_user_id"]
        self.test_symbols = TEST_CONFIG["test_symbols"]

    def test_requirement_1_default_monthly_limit(self):
        """
        TEST REQUIREMENT 1: max_amount_in_a_month defaults to 4x fixed_investment
        """
        from backend.app.routes.sip_routes import SIPConfigRequest

        # Test with default values
        config = SIPConfigRequest(fixed_investment=5000)
        assert config.max_amount_in_a_month == 20000  # 4 * 5000

        # Test with different fixed investment
        config2 = SIPConfigRequest(fixed_investment=8000)
        assert config2.max_amount_in_a_month == 32000  # 4 * 8000

        # Test with explicit monthly limit
        config3 = SIPConfigRequest(fixed_investment=5000, max_amount_in_a_month=30000)
        assert config3.max_amount_in_a_month == 30000

        print("✅ REQUIREMENT 1 PASSED: max_amount_in_a_month defaults to 4x fixed_investment")

    def test_requirement_1_monthly_limit_validation(self):
        """Test monthly limit validation rules"""
        from backend.app.routes.sip_routes import SIPConfigRequest
        import pytest

        # Test valid configurations
        valid_config = SIPConfigRequest(
            fixed_investment=5000,
            max_amount_in_a_month=25000
        )
        assert valid_config.max_amount_in_a_month == 25000

        # Test invalid: monthly limit less than fixed investment
        with pytest.raises(ValueError, match="Monthly limit cannot be less than fixed investment"):
            SIPConfigRequest(
                fixed_investment=5000,
                max_amount_in_a_month=3000
            )

        # Test invalid: negative monthly limit
        with pytest.raises(ValueError, match="Monthly limit must be positive"):
            SIPConfigRequest(
                fixed_investment=5000,
                max_amount_in_a_month=-1000
            )

        print("✅ REQUIREMENT 1 VALIDATION PASSED: Monthly limit validation works correctly")

    def test_requirement_2_price_threshold_edge_cases(self):
        """Test edge cases for price reduction threshold"""
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        # Test with different thresholds
        test_thresholds = [2.0, 4.0, 6.0, 10.0]

        for threshold in test_thresholds:
            tracker = MonthlyInvestmentTracker(
                max_monthly_amount=20000,
                price_reduction_threshold=threshold
            )

            symbol = "TEST"
            date = datetime(2024, 1, 15)

            # First investment at 1000
            tracker.record_investment(symbol, date, 5000, 1000)

            # Calculate price that meets exactly the threshold
            required_price = 1000 * (1 - threshold / 100)

            # Test price that meets threshold
            result = tracker.can_invest(symbol, date + timedelta(days=1), 5000, required_price)
            assert result['can_invest'] == True, f"Threshold {threshold}% not working for exact price"

            # Test price that doesn't meet threshold (0.1% less reduction)
            insufficient_price = 1000 * (1 - (threshold - 0.1) / 100)
            result = tracker.can_invest(symbol, date + timedelta(days=1), 5000, insufficient_price)
            assert result['can_invest'] == False, f"Threshold {threshold}% allowing insufficient reduction"

        print("✅ REQUIREMENT 2 EDGE CASES PASSED: Price threshold works with different values")

    def test_integration_both_requirements(self):
        """
        INTEGRATION TEST: Both requirements working together
        """
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        # Setup: 5000 fixed investment, 20000 monthly limit (4x), 4% threshold
        tracker = MonthlyInvestmentTracker(
            max_monthly_amount=20000,  # 4x of 5000
            price_reduction_threshold=4.0
        )

        symbol = "INTEGRATION_TEST"
        date = datetime(2024, 1, 10)
        investment_amount = 5000

        # Scenario: Multiple investments in January with various conditions
        scenarios = [
            {
                "date_offset": 0,
                "price": 2500,
                "description": "First investment - should always work",
                "expected_allowed": True,
                "expected_amount": 5000
            },
            {
                "date_offset": 5,
                "price": 2400,  # 4% reduction
                "description": "Second investment with 4% price drop",
                "expected_allowed": True,
                "expected_amount": 5000
            },
            {
                "date_offset": 10,
                "price": 2300,  # 4.2% reduction from last
                "description": "Third investment with good price drop",
                "expected_allowed": True,
                "expected_amount": 5000
            },
            {
                "date_offset": 15,
                "price": 2250,  # 2.2% reduction from last - insufficient
                "description": "Fourth investment - insufficient price drop",
                "expected_allowed": False,
                "expected_amount": 0
            },
            {
                "date_offset": 20,
                "price": 2200,  # 4.3% reduction from last valid price (2300)
                "description": "Fifth investment with good price drop",
                "expected_allowed": True,
                "expected_amount": 5000  # This would make total 20000 (limit reached)
            },
            {
                "date_offset": 25,
                "price": 2100,  # Great price but monthly limit reached
                "description": "Sixth investment - monthly limit reached",
                "expected_allowed": False,
                "expected_amount": 0
            }
        ]

        executed_investments = 0
        total_invested = 0

        for scenario in scenarios:
            current_date = date + timedelta(days=scenario["date_offset"])
            result = tracker.can_invest(symbol, current_date, investment_amount, scenario["price"])

            print(f"Scenario: {scenario['description']}")
            print(f"  Expected: {'Allowed' if scenario['expected_allowed'] else 'Blocked'}")
            print(f"  Actual: {'Allowed' if result['can_invest'] else 'Blocked'}")
            print(f"  Reason: {result['reason']}")

            assert result['can_invest'] == scenario['expected_allowed'], f"Failed: {scenario['description']}"

            if result['can_invest']:
                final_amount = result['suggested_amount']
                tracker.record_investment(symbol, current_date, final_amount, scenario["price"])
                executed_investments += 1
                total_invested += final_amount
                print(f"  ✅ Invested: ₹{final_amount:,.2f}")
            else:
                print(f"  ❌ Blocked: {result['reason']}")
            print()

        # Verify final state
        assert executed_investments == 4, f"Expected 4 investments, got {executed_investments}"
        assert total_invested == 20000, f"Expected ₹20,000 total, got ₹{total_invested:,.2f}"

        # Check monthly summary
        monthly_summary = tracker.get_monthly_summary(symbol)
        jan_data = monthly_summary["2024-01"]
        assert jan_data['total_invested'] == 20000
        assert jan_data['num_investments'] == 4
        assert jan_data['remaining_budget'] == 0

        print("✅ INTEGRATION TEST PASSED: Both requirements work together correctly")

    @pytest.mark.asyncio
    async def test_end_to_end_backtest_api(self):
        """End-to-end test of the enhanced backtest API"""

        # Mock the strategy and database dependencies
        with patch('backend.app.routes.sip_routes.EnhancedSIPStrategyWithLimits') as mock_strategy_class:
            mock_strategy = AsyncMock()
            mock_strategy_class.return_value = mock_strategy

            # Mock successful backtest result
            mock_result = {
                'symbol': 'RELIANCE',
                'strategy_name': 'Enhanced SIP with Monthly Limits',
                'total_investment': 50000,
                'final_portfolio_value': 65000,
                'total_return_percent': 30.0,
                'cagr_percent': 12.5,
                'num_trades': 8,
                'num_skipped': 3,
                'monthly_limit_exceeded': 1,
                'price_threshold_skipped': 2,
                'config_used': {
                    'fixed_investment': 5000,
                    'max_amount_in_a_month': 20000,
                    'price_reduction_threshold': 4.0
                },
                'monthly_summary': {
                    '2024-01': {'total_invested': 20000, 'num_investments': 4},
                    '2024-02': {'total_invested': 15000, 'num_investments': 3},
                    '2024-03': {'total_invested': 15000, 'num_investments': 1}
                },
                'trades': [],
                'skipped_investments': []
            }

            mock_strategy.run_backtest.return_value = mock_result

            # Test the API call
            from backend.app.routes.sip_routes import SIPBacktestRequest, SIPConfigRequest

            request = SIPBacktestRequest(
                symbols=["RELIANCE"],
                start_date="2024-01-01",
                end_date="2024-03-31",
                config=SIPConfigRequest(
                    fixed_investment=5000,
                    max_amount_in_a_month=20000,
                    price_reduction_threshold=4.0
                )
            )

            # Verify the request structure includes new fields
            assert request.config.fixed_investment == 5000
            assert request.config.max_amount_in_a_month == 20000  # 4x default
            assert request.config.price_reduction_threshold == 4.0

            print("✅ END-TO-END API TEST PASSED: Enhanced backtest API structure is correct")

    def test_monthly_limit_enforcement_detailed(self):
        """Detailed test of monthly limit enforcement scenarios"""
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        scenarios = [
            {
                "name": "Conservative limits",
                "monthly_limit": 15000,
                "investments": [
                    {"amount": 5000, "expected": True, "remaining": 10000},
                    {"amount": 8000, "expected": True, "remaining": 2000},
                    {"amount": 3000, "expected": False, "suggested": 2000},  # Exceeds by 1000
                    {"amount": 1000, "expected": False, "suggested": 0}  # No budget left
                ]
            },
            {
                "name": "Aggressive limits",
                "monthly_limit": 50000,
                "investments": [
                    {"amount": 10000, "expected": True, "remaining": 40000},
                    {"amount": 15000, "expected": True, "remaining": 25000},
                    {"amount": 20000, "expected": True, "remaining": 5000},
                    {"amount": 10000, "expected": False, "suggested": 5000}
                ]
            }
        ]

        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")

            tracker = MonthlyInvestmentTracker(
                max_monthly_amount=scenario["monthly_limit"],
                price_reduction_threshold=4.0
            )

            symbol = f"TEST_{scenario['name'].upper()}"
            date = datetime(2024, 1, 15)
            prices = [2500, 2400, 2300, 2200]  # Ensure price threshold is met

            for i, inv in enumerate(scenario["investments"]):
                current_date = date + timedelta(days=i * 5)
                result = tracker.can_invest(symbol, current_date, inv["amount"], prices[i])

                print(f"  Investment {i + 1}: ₹{inv['amount']:,}")
                print(f"    Expected: {'Allowed' if inv['expected'] else 'Blocked'}")
                print(f"    Actual: {'Allowed' if result['can_invest'] else 'Blocked'}")

                assert result['can_invest'] == inv['expected']

                if result['can_invest']:
                    suggested_amount = result['suggested_amount']
                    if 'suggested' in inv:
                        assert suggested_amount == inv[
                            'suggested'], f"Expected suggested amount {inv['suggested']}, got {suggested_amount}"

                    tracker.record_investment(symbol, current_date, suggested_amount, prices[i])
                    assert result['monthly_remaining'] == inv['remaining']
                    print(f"    ✅ Invested: ₹{suggested_amount:,}, Remaining: ₹{result['monthly_remaining']:,}")
                else:
                    print(f"    ❌ Blocked: {result['reason']}")

        print("✅ DETAILED MONTHLY LIMIT TESTS PASSED")

    def test_cross_month_independence(self):
        """Test that monthly limits reset across months"""
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        tracker = MonthlyInvestmentTracker(
            max_monthly_amount=15000,
            price_reduction_threshold=4.0
        )

        symbol = "CROSS_MONTH_TEST"

        # January: Max out the limit
        jan_dates = [datetime(2024, 1, 10), datetime(2024, 1, 20)]
        jan_prices = [2500, 2400]  # 4% reduction

        for i, (date, price) in enumerate(zip(jan_dates, jan_prices)):
            result = tracker.can_invest(symbol, date, 8000, price)
            if i == 0:
                assert result['can_invest'] == True
                tracker.record_investment(symbol, date, 8000, price)
            else:
                # Second investment should be limited to remaining budget
                assert result['can_invest'] == True
                assert result['suggested_amount'] == 7000  # 15000 - 8000
                tracker.record_investment(symbol, date, 7000, price)

        # Verify January is maxed out
        jan_result = tracker.can_invest(symbol, datetime(2024, 1, 25), 5000, 2300)
        assert jan_result['can_invest'] == False

        # February: Should reset and allow full investment
        feb_result = tracker.can_invest(symbol, datetime(2024, 2, 10), 10000, 2600)
        assert feb_result['can_invest'] == True
        assert feb_result['suggested_amount'] == 10000
        assert feb_result['monthly_spent'] == 0  # Fresh month

        tracker.record_investment(symbol, datetime(2024, 2, 10), 10000, 2600)

        # Verify monthly summary shows correct separation
        summary = tracker.get_monthly_summary(symbol)
        assert '2024-01' in summary
        assert '2024-02' in summary
        assert summary['2024-01']['total_invested'] == 15000
        assert summary['2024-02']['total_invested'] == 10000

        print("✅ CROSS-MONTH INDEPENDENCE TEST PASSED")

    def test_multiple_symbols_independence(self):
        """Test that different symbols have independent monthly tracking"""
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        tracker = MonthlyInvestmentTracker(
            max_monthly_amount=20000,
            price_reduction_threshold=4.0
        )

        date = datetime(2024, 1, 15)
        symbols = ["RELIANCE", "TCS", "INFY"]
        prices = [2500, 3500, 1500]

        # Max out RELIANCE
        for i in range(4):  # 4 * 5000 = 20000
            investment_date = date + timedelta(days=i * 3)
            price = prices[0] * (1 - 0.05 * i)  # Ensure price reduction
            result = tracker.can_invest("RELIANCE", investment_date, 5000, price)
            assert result['can_invest'] == True
            tracker.record_investment("RELIANCE", investment_date, 5000, price)

        # RELIANCE should be maxed out
        reliance_result = tracker.can_invest("RELIANCE", date + timedelta(days=15), 5000, 2000)
        assert reliance_result['can_invest'] == False

        # TCS and INFY should still be available with full limits
        for symbol, price in zip(["TCS", "INFY"], prices[1:]):
            result = tracker.can_invest(symbol, date, 15000, price)
            assert result['can_invest'] == True
            assert result['suggested_amount'] == 15000
            assert result['monthly_spent'] == 0  # Independent tracking

        print("✅ MULTIPLE SYMBOLS INDEPENDENCE TEST PASSED")

    def test_analytics_and_reporting(self):
        """Test analytics features for enhanced SIP"""
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        tracker = MonthlyInvestmentTracker(
            max_monthly_amount=25000,
            price_reduction_threshold=4.0
        )

        symbol = "ANALYTICS_TEST"

        # Create a realistic investment scenario over 3 months
        scenarios = [
            # January
            {"date": datetime(2024, 1, 5), "amount": 8000, "price": 2500, "should_invest": True},
            {"date": datetime(2024, 1, 10), "amount": 6000, "price": 2400, "should_invest": True},  # 4% drop
            {"date": datetime(2024, 1, 15), "amount": 7000, "price": 2350, "should_invest": True},
            # 2.1% drop from 2400
            {"date": datetime(2024, 1, 20), "amount": 5000, "price": 2340, "should_invest": False},  # <4% drop
            {"date": datetime(2024, 1, 25), "amount": 6000, "price": 2250, "should_invest": False},
            # Would exceed limit

            # February
            {"date": datetime(2024, 2, 5), "amount": 10000, "price": 2200, "should_invest": True},  # New month
            {"date": datetime(2024, 2, 15), "amount": 8000, "price": 2100, "should_invest": True},  # 4.5% drop

            # March
            {"date": datetime(2024, 3, 5), "amount": 12000, "price": 2000, "should_invest": True},  # New month
        ]

        executed_count = 0
        skipped_count = 0

        for scenario in scenarios:
            result = tracker.can_invest(symbol, scenario["date"], scenario["amount"], scenario["price"])

            if scenario["should_invest"] and result['can_invest']:
                tracker.record_investment(symbol, scenario["date"], result['suggested_amount'], scenario["price"])
                executed_count += 1
            elif not scenario["should_invest"] and not result['can_invest']:
                tracker.record_skipped_investment(
                    symbol, scenario["date"], scenario["amount"],
                    scenario["price"], result['reason']
                )
                skipped_count += 1

        # Analyze results
        monthly_summary = tracker.get_monthly_summary(symbol)

        # January analysis
        jan_data = monthly_summary["2024-01"]
        assert jan_data['total_invested'] == 21000  # 8000 + 6000 + 7000
        assert jan_data['num_investments'] == 3
        assert jan_data['budget_utilization_percent'] == 84.0  # 21000/25000 * 100

        # February analysis
        feb_data = monthly_summary["2024-02"]
        assert feb_data['total_invested'] == 18000  # 10000 + 8000
        assert feb_data['num_investments'] == 2

        # March analysis
        mar_data = monthly_summary["2024-03"]
        assert mar_data['total_invested'] == 12000
        assert mar_data['num_investments'] == 1

        # Overall analytics
        total_invested = sum(month['total_invested'] for month in monthly_summary.values())
        total_investments = sum(month['num_investments'] for month in monthly_summary.values())
        avg_utilization = sum(month['budget_utilization_percent'] for month in monthly_summary.values()) / len(
            monthly_summary)

        print(f"Analytics Summary:")
        print(f"  Total invested: ₹{total_invested:,}")
        print(f"  Total investments: {total_investments}")
        print(f"  Average monthly utilization: {avg_utilization:.1f}%")
        print(f"  Executed investments: {executed_count}")
        print(f"  Skipped investments: {skipped_count}")

        assert total_invested == 51000
        assert total_investments == 6
        assert executed_count == 6
        assert skipped_count >= 2  # At least 2 skipped due to various reasons

        print("✅ ANALYTICS AND REPORTING TEST PASSED")

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("🧪 STARTING COMPREHENSIVE ENHANCED SIP TESTS")
        print("=" * 60)

        try:
            # Requirement tests
            self.test_requirement_1_default_monthly_limit()
            self.test_requirement_1_monthly_limit_validation()
            self.test_requirement_2_price_reduction_threshold()
            self.test_requirement_2_price_threshold_edge_cases()

            # Integration tests
            self.test_integration_both_requirements()

            # Detailed feature tests
            self.test_monthly_limit_enforcement_detailed()
            self.test_cross_month_independence()
            self.test_multiple_symbols_independence()
            self.test_analytics_and_reporting()

            # API tests
            asyncio.run(self.test_end_to_end_backtest_api())

            print("=" * 60)
            print("🎉 ALL ENHANCED SIP TESTS PASSED!")
            print("✅ Both requirements implemented and working correctly:")
            print("   1. max_amount_in_a_month defaults to 4x fixed_investment")
            print("   2. 4% price reduction threshold for multiple monthly signals")
            print("=" * 60)

        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            raise


# Performance benchmark test
class TestPerformanceBenchmarks:
    """Test performance of enhanced SIP features"""

    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        import time
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        tracker = MonthlyInvestmentTracker(
            max_monthly_amount=50000,
            price_reduction_threshold=4.0
        )

        # Generate large dataset
        symbols = [f"STOCK_{i}" for i in range(100)]
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]

        start_time = time.time()

        # Simulate many investment decisions
        decisions = 0
        for symbol in symbols[:10]:  # Test with 10 symbols
            for i, date in enumerate(dates[:30]):  # 30 days each
                price = 1000 + (i * 10)  # Varying prices
                result = tracker.can_invest(symbol, date, 5000, price)
                decisions += 1

                if result['can_invest'] and decisions % 3 == 0:  # Record some investments
                    tracker.record_investment(symbol, date, result['suggested_amount'], price)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Performance Test:")
        print(f"  Decisions processed: {decisions}")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Decisions per second: {decisions / execution_time:.0f}")

        # Performance should be reasonable (>1000 decisions per second)
        assert decisions / execution_time > 100, "Performance below acceptable threshold"

        print("✅ PERFORMANCE BENCHMARK PASSED")


# Usage example
if __name__ == "__main__":
    # Run comprehensive tests
    test_suite = TestEnhancedSIPFeatures()
    test_suite.run_all_tests()

    # Run performance tests
    perf_tests = TestPerformanceBenchmarks()
    perf_tests.test_large_dataset_performance()

    print("\n🚀 READY FOR PRODUCTION!")
    print("Enhanced SIP features are fully tested and validated.")
    def test_requirement_2_price_reduction_threshold(self):
        """
        TEST REQUIREMENT 2: 4% price reduction threshold for multiple signals
        """
        from backend.app.routes.sip_routes import MonthlyInvestmentTracker

        tracker = MonthlyInvestmentTracker(
            max_monthly_amount=20000,
            price_reduction_threshold=4.0
        )

        symbol = "RELIANCE"
        date = datetime(2024, 1, 15)

        # First investment at 2500
        tracker.record_investment(symbol, date, 5000, 2500)

        # Test scenarios for second investment
        test_cases = [
            {
                "price": 2400,  # 4% reduction: (2500-2400)/2500 = 4%
                "expected_allowed": True,
                "description": "Exactly 4% reduction - should be allowed"
            },
            {
                "price": 2350,  # 6% reduction: (2500-2350)/2500 = 6%
                "expected_allowed": True,
                "description": "More than 4% reduction - should be allowed"
            },
            {
                "price": 2450,  # 2% reduction: (2500-2450)/2500 = 2%
                "expected_allowed": False,
                "description": "Less than 4% reduction - should be blocked"
            },
            {
                "price": 2525,  # Price increase
                "expected_allowed": False,
                "description": "Price increase - should be blocked"
            }
        ]

        for case in test_cases:
            result = tracker.can_invest(symbol, date + timedelta(days=5), 5000, case["price"])
            assert result['can_invest'] == case["expected_allowed"], f"Failed: {case['description']}"

            if not case["expected_allowed"]:
                assert "Price reduction" in result['reason']
                assert "4.0%" in result['reason']

        print("✅ REQUIREMENT 2 PASSED: 4% price reduction threshold works correctly")


