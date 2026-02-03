"""
Real-Time Document Intelligence & Compliance Engine
Financial Risk Assessment System
Version: 1.0
Date: 2026-02-03

This module provides comprehensive financial risk assessment (Feature 2) for documents
including contract values, payment terms, financial exposure, and market risk analysis.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import uuid
from .compliance_risk import RiskLevel, RiskCategory, RiskFactor, RiskAssessmentResult

logger = logging.getLogger(__name__)


class FinancialRiskType(Enum):
    """Types of financial risks"""

    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    MARKET_RISK = "market_risk"
    OPERATIONAL_RISK = "operational_risk"
    CURRENCY_RISK = "currency_risk"
    COUNTERPARTY_RISK = "counterparty_risk"


@dataclass
class FinancialMetrics:
    """Financial metrics extracted from document"""

    total_value: Optional[float] = None
    payment_terms_days: Optional[int] = None
    penalties: List[float] = None
    interest_rates: List[float] = None
    currency: str = "USD"
    payment_schedule: List[Tuple[datetime, float]] = None
    collateral_value: Optional[float] = None
    guarantees: List[str] = None


class FinancialRiskAssessor:
    """Financial risk assessment system (Feature 2)"""

    def __init__(
        self,
        currency_rates: Dict[str, float] = None,
        market_volatility: Dict[str, float] = None,
        credit_ratings: Dict[str, str] = None,
    ):
        """
        Initialize financial risk assessor

        Args:
            currency_rates: Exchange rates for currency conversion
            market_volatility: Market volatility indicators
            credit_ratings: Credit ratings for counterparties
        """
        self.currency_rates = currency_rates or {
            "USD": 1.0,
            "EUR": 0.85,
            "GBP": 0.73,
            "JPY": 110.0,
        }
        self.market_volatility = market_volatility or {
            "equity": 0.2,
            "bond": 0.05,
            "commodity": 0.3,
            "forex": 0.15,
        }
        self.credit_ratings = credit_ratings or {}

        # Risk parameters
        self.risk_thresholds = {
            "minimal": 0.2,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "critical": 1.0,
        }

        # Financial exposure limits
        self.exposure_limits = {
            "single_counterparty": 0.1,  # 10% of portfolio
            "currency_exposure": 0.2,  # 20% in foreign currency
            "sector_concentration": 0.15,  # 15% in single sector
        }

        # Historical financial data
        self.financial_history: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = {}

    def assess_financial_risk(
        self,
        document_data: Dict[str, Any],
        extracted_entities: Dict[str, Any] = None,
        market_context: Dict[str, Any] = None,
    ) -> RiskAssessmentResult:
        """
        Assess financial risk for document

        Args:
            document_data: Document analysis data
            extracted_entities: Extracted financial entities
            market_context: Current market conditions

        Returns:
            Financial risk assessment result
        """
        try:
            current_time = datetime.now()
            document_id = document_data.get("document_id", "unknown")

            # Extract financial metrics
            financial_metrics = self._extract_financial_metrics(
                document_data, extracted_entities
            )

            risk_factors = []

            # 1. Credit risk assessment
            credit_risk = self._assess_credit_risk(financial_metrics, document_data)
            if credit_risk:
                risk_factors.append(credit_risk)

            # 2. Liquidity risk assessment
            liquidity_risk = self._assess_liquidity_risk(
                financial_metrics, document_data
            )
            if liquidity_risk:
                risk_factors.append(liquidity_risk)

            # 3. Market risk assessment
            market_risk = self._assess_market_risk(financial_metrics, market_context)
            if market_risk:
                risk_factors.append(market_risk)

            # 4. Currency risk assessment
            currency_risk = self._assess_currency_risk(financial_metrics)
            if currency_risk:
                risk_factors.append(currency_risk)

            # 5. Counterparty risk assessment
            counterparty_risk = self._assess_counterparty_risk(
                financial_metrics, document_data
            )
            if counterparty_risk:
                risk_factors.append(counterparty_risk)

            # 6. Operational financial risk
            operational_risk = self._assess_operational_financial_risk(document_data)
            if operational_risk:
                risk_factors.append(operational_risk)

            # Calculate overall financial risk score
            risk_score = self._calculate_financial_risk_score(
                risk_factors, financial_metrics
            )
            risk_level = self._get_risk_level(risk_score)

            # Calculate confidence
            confidence_score = self._calculate_financial_confidence(
                risk_factors, financial_metrics
            )

            # Business impact assessment
            business_impact = self._assess_financial_business_impact(
                risk_score, financial_metrics
            )

            # Generate mitigation recommendations
            recommendations = self._generate_financial_recommendations(
                risk_factors, financial_metrics
            )

            result = RiskAssessmentResult(
                assessment_id=str(uuid.uuid4()),
                document_id=document_id,
                risk_category=RiskCategory.FINANCIAL,
                risk_level=risk_level,
                risk_score=risk_score,
                confidence_score=confidence_score,
                assessed_at=current_time,
                risk_factors=risk_factors,
                business_impact=business_impact,
                mitigation_recommendations=recommendations,
                context={
                    "financial_metrics": financial_metrics.__dict__
                    if financial_metrics
                    else {},
                    "total_exposure": financial_metrics.total_value
                    if financial_metrics
                    else 0,
                    "currency": financial_metrics.currency
                    if financial_metrics
                    else "USD",
                },
            )

            # Store in financial history
            if document_id and financial_metrics:
                self._store_financial_history(
                    document_id, financial_metrics, current_time
                )

            return result

        except Exception as e:
            logger.error(f"Financial risk assessment failed: {e}")
            return self._create_default_financial_assessment(document_data)

    def _extract_financial_metrics(
        self, document_data: Dict[str, Any], extracted_entities: Dict[str, Any] = None
    ) -> FinancialMetrics:
        """Extract financial metrics from document"""
        try:
            metrics = FinancialMetrics()

            # Extract from document data
            if "financial_data" in document_data:
                fin_data = document_data["financial_data"]
                metrics.total_value = fin_data.get("total_value")
                metrics.currency = fin_data.get("currency", "USD")
                metrics.collateral_value = fin_data.get("collateral_value")

            # Extract from entities if available
            if extracted_entities:
                # Extract monetary amounts
                money_entities = extracted_entities.get("MONEY", [])
                if money_entities:
                    amounts = []
                    for entity in money_entities:
                        amount_text = entity.get("text", "")
                        amount = self._parse_monetary_amount(amount_text)
                        if amount:
                            amounts.append(amount)

                    if amounts:
                        metrics.total_value = max(
                            amounts
                        )  # Use largest amount as total value

                # Extract payment terms
                date_entities = extracted_entities.get("DATE", [])
                if date_entities:
                    payment_dates = []
                    for entity in date_entities:
                        date_text = entity.get("text", "")
                        parsed_date = self._parse_payment_date(date_text)
                        if parsed_date:
                            payment_dates.append(parsed_date)

                    if payment_dates:
                        # Calculate payment terms in days
                        earliest_date = min(payment_dates)
                        latest_date = max(payment_dates)
                        metrics.payment_terms_days = (latest_date - earliest_date).days

                # Extract interest rates and penalties
                percent_entities = extracted_entities.get("PERCENT", [])
                if percent_entities:
                    rates = []
                    penalties = []

                    for entity in percent_entities:
                        rate_text = entity.get("text", "").lower()
                        rate_value = self._parse_percentage(entity.get("text", ""))

                        if rate_value is not None:
                            if any(
                                word in rate_text
                                for word in ["interest", "rate", "apr"]
                            ):
                                rates.append(rate_value)
                            elif any(
                                word in rate_text for word in ["penalty", "fee", "late"]
                            ):
                                penalties.append(rate_value)

                    metrics.interest_rates = rates if rates else None
                    metrics.penalties = penalties if penalties else None

            return metrics

        except Exception as e:
            logger.warning(f"Financial metrics extraction failed: {e}")
            return FinancialMetrics()

    def _parse_monetary_amount(self, amount_text: str) -> Optional[float]:
        """Parse monetary amount from text"""
        try:
            # Remove currency symbols and commas
            clean_text = re.sub(r"[£$€¥,]", "", amount_text.strip())

            # Handle millions/billions notation
            if "million" in clean_text.lower() or "m" in clean_text.lower():
                multiplier = 1_000_000
                clean_text = re.sub(r"million|m", "", clean_text, flags=re.IGNORECASE)
            elif "billion" in clean_text.lower() or "b" in clean_text.lower():
                multiplier = 1_000_000_000
                clean_text = re.sub(r"billion|b", "", clean_text, flags=re.IGNORECASE)
            else:
                multiplier = 1

            # Extract number
            number_match = re.search(r"[\d.]+", clean_text)
            if number_match:
                return float(number_match.group()) * multiplier

            return None

        except Exception:
            return None

    def _parse_payment_date(self, date_text: str) -> Optional[datetime]:
        """Parse payment date from text"""
        try:
            # This is a simplified parser - in practice, you'd use a more robust date parser
            from dateutil.parser import parse

            return parse(date_text, fuzzy=True)
        except Exception:
            return None

    def _parse_percentage(self, percent_text: str) -> Optional[float]:
        """Parse percentage from text"""
        try:
            # Extract numeric value
            number_match = re.search(r"([\d.]+)", percent_text)
            if number_match:
                return float(number_match.group(1))
            return None
        except Exception:
            return None

    def _assess_credit_risk(
        self, metrics: FinancialMetrics, document_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess credit risk"""
        try:
            credit_risk_score = 0.0
            evidence = []

            # Check counterparty credit rating
            counterparty = document_data.get("counterparty", {})
            counterparty_name = counterparty.get("name", "")

            if counterparty_name in self.credit_ratings:
                rating = self.credit_ratings[counterparty_name]
                # Convert rating to risk score (AAA=0.1, BB=0.7, D=1.0)
                rating_risk_map = {
                    "AAA": 0.1,
                    "AA": 0.15,
                    "A": 0.2,
                    "BBB": 0.3,
                    "BB": 0.5,
                    "B": 0.7,
                    "CCC": 0.8,
                    "CC": 0.9,
                    "C": 0.95,
                    "D": 1.0,
                }
                credit_risk_score = rating_risk_map.get(rating.upper(), 0.5)
                evidence.append(f"Credit rating: {rating}")
            else:
                credit_risk_score = 0.6  # Unknown counterparty - medium risk
                evidence.append("No credit rating available")

            # Adjust for payment terms
            if metrics.payment_terms_days:
                if metrics.payment_terms_days > 90:
                    credit_risk_score += 0.2
                    evidence.append(
                        f"Extended payment terms: {metrics.payment_terms_days} days"
                    )
                elif metrics.payment_terms_days > 60:
                    credit_risk_score += 0.1
                    evidence.append(
                        f"Long payment terms: {metrics.payment_terms_days} days"
                    )

            # Check for guarantees or collateral
            if metrics.collateral_value and metrics.total_value:
                collateral_ratio = metrics.collateral_value / metrics.total_value
                if collateral_ratio > 0.8:
                    credit_risk_score *= 0.7  # Well-collateralized
                    evidence.append(f"High collateral ratio: {collateral_ratio:.1%}")
                elif collateral_ratio > 0.5:
                    credit_risk_score *= 0.85
                    evidence.append(f"Moderate collateral: {collateral_ratio:.1%}")

            return RiskFactor(
                factor_id="credit_risk",
                name="Credit Risk",
                description="Risk of counterparty default or payment failure",
                weight=0.3,
                value=min(1.0, credit_risk_score),
                confidence=0.8 if counterparty_name in self.credit_ratings else 0.6,
                evidence=evidence,
                source="credit_assessment",
            )

        except Exception as e:
            logger.warning(f"Credit risk assessment failed: {e}")
            return None

    def _assess_liquidity_risk(
        self, metrics: FinancialMetrics, document_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess liquidity risk"""
        try:
            liquidity_risk_score = 0.0
            evidence = []

            # Check payment schedule concentration
            if metrics.payment_schedule:
                # Analyze payment concentration in time periods
                total_payments = sum(amount for _, amount in metrics.payment_schedule)
                near_term_payments = sum(
                    amount
                    for date, amount in metrics.payment_schedule
                    if (date - datetime.now()).days <= 30
                )

                if total_payments > 0:
                    near_term_ratio = near_term_payments / total_payments
                    if near_term_ratio > 0.5:
                        liquidity_risk_score += 0.4
                        evidence.append(
                            f"High near-term payment concentration: {near_term_ratio:.1%}"
                        )
                    elif near_term_ratio > 0.3:
                        liquidity_risk_score += 0.2
                        evidence.append(
                            f"Moderate near-term payments: {near_term_ratio:.1%}"
                        )

            # Check contract value vs. typical exposure
            if metrics.total_value:
                if metrics.total_value > 10_000_000:  # Large exposure
                    liquidity_risk_score += 0.3
                    evidence.append(
                        f"Large financial exposure: ${metrics.total_value:,.0f}"
                    )
                elif metrics.total_value > 1_000_000:
                    liquidity_risk_score += 0.1
                    evidence.append(
                        f"Significant exposure: ${metrics.total_value:,.0f}"
                    )

            # Check payment terms impact on cash flow
            if metrics.payment_terms_days:
                if metrics.payment_terms_days < 15:  # Very short terms
                    liquidity_risk_score += 0.2
                    evidence.append(
                        f"Short payment terms impact: {metrics.payment_terms_days} days"
                    )

            # Market liquidity considerations
            market_data = document_data.get("market_context", {})
            market_liquidity = market_data.get("liquidity_index", 1.0)
            if market_liquidity < 0.7:
                liquidity_risk_score += 0.2
                evidence.append(f"Poor market liquidity: {market_liquidity:.2f}")

            return RiskFactor(
                factor_id="liquidity_risk",
                name="Liquidity Risk",
                description="Risk of cash flow constraints or funding difficulties",
                weight=0.2,
                value=min(1.0, liquidity_risk_score),
                confidence=0.7,
                evidence=evidence,
                source="liquidity_assessment",
            )

        except Exception as e:
            logger.warning(f"Liquidity risk assessment failed: {e}")
            return None

    def _assess_market_risk(
        self, metrics: FinancialMetrics, market_context: Dict[str, Any] = None
    ) -> Optional[RiskFactor]:
        """Assess market risk"""
        try:
            market_risk_score = 0.0
            evidence = []

            # Base market risk from volatility
            if market_context:
                market_vol = market_context.get("volatility", 0.2)
                market_risk_score += market_vol * 2  # Scale volatility to risk score
                evidence.append(f"Market volatility: {market_vol:.1%}")

                # Interest rate risk
                interest_rate_change = market_context.get("interest_rate_delta", 0)
                if abs(interest_rate_change) > 0.02:  # 2% change
                    market_risk_score += 0.2
                    evidence.append(f"Interest rate change: {interest_rate_change:.1%}")
            else:
                market_risk_score = 0.3  # Default market risk
                evidence.append("No market context available")

            # Contract-specific market risk
            if metrics.interest_rates:
                avg_rate = np.mean(metrics.interest_rates)
                if avg_rate > 8.0:  # High interest rate
                    market_risk_score += 0.2
                    evidence.append(f"High interest rate exposure: {avg_rate:.1%}")
                elif avg_rate > 5.0:
                    market_risk_score += 0.1
                    evidence.append(f"Moderate rate exposure: {avg_rate:.1%}")

            # Currency market risk (handled separately in currency risk)

            return RiskFactor(
                factor_id="market_risk",
                name="Market Risk",
                description="Risk from market volatility and economic conditions",
                weight=0.15,
                value=min(1.0, market_risk_score),
                confidence=0.8 if market_context else 0.5,
                evidence=evidence,
                source="market_assessment",
            )

        except Exception as e:
            logger.warning(f"Market risk assessment failed: {e}")
            return None

    def _assess_currency_risk(self, metrics: FinancialMetrics) -> Optional[RiskFactor]:
        """Assess currency risk"""
        try:
            if not metrics.currency or metrics.currency == "USD":
                return None  # No currency risk for base currency

            currency_risk_score = 0.0
            evidence = []

            # Base currency risk from volatility
            currency_volatility = self.market_volatility.get("forex", 0.15)
            currency_risk_score += currency_volatility * 2
            evidence.append(
                f"Currency volatility ({metrics.currency}): {currency_volatility:.1%}"
            )

            # Exchange rate risk
            current_rate = self.currency_rates.get(metrics.currency, 1.0)
            if current_rate != 1.0:
                # Simplified risk based on deviation from par
                rate_deviation = abs(1.0 - current_rate) / max(1.0, current_rate)
                currency_risk_score += rate_deviation * 0.5
                evidence.append(f"Exchange rate ({metrics.currency}): {current_rate}")

            # Exposure size impact
            if metrics.total_value:
                # Convert to USD for comparison
                usd_exposure = metrics.total_value * current_rate
                if usd_exposure > 5_000_000:
                    currency_risk_score += 0.2
                    evidence.append(
                        f"Large foreign currency exposure: ${usd_exposure:,.0f}"
                    )
                elif usd_exposure > 1_000_000:
                    currency_risk_score += 0.1
                    evidence.append(f"Significant FX exposure: ${usd_exposure:,.0f}")

            return RiskFactor(
                factor_id="currency_risk",
                name="Currency Risk",
                description="Risk from foreign exchange rate fluctuations",
                weight=0.1,
                value=min(1.0, currency_risk_score),
                confidence=0.7,
                evidence=evidence,
                source="currency_assessment",
            )

        except Exception as e:
            logger.warning(f"Currency risk assessment failed: {e}")
            return None

    def _assess_counterparty_risk(
        self, metrics: FinancialMetrics, document_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess counterparty risk"""
        try:
            counterparty_risk_score = 0.0
            evidence = []

            counterparty = document_data.get("counterparty", {})

            # Industry risk
            industry = counterparty.get("industry", "")
            high_risk_industries = ["crypto", "gambling", "commodities", "startups"]
            if any(risk_ind in industry.lower() for risk_ind in high_risk_industries):
                counterparty_risk_score += 0.3
                evidence.append(f"High-risk industry: {industry}")

            # Geographic risk
            jurisdiction = counterparty.get("jurisdiction", "")
            high_risk_jurisdictions = ["emerging", "offshore", "sanctioned"]
            if any(
                risk_jur in jurisdiction.lower() for risk_jur in high_risk_jurisdictions
            ):
                counterparty_risk_score += 0.2
                evidence.append(f"High-risk jurisdiction: {jurisdiction}")

            # Company size/stability
            company_size = counterparty.get("size", "")
            if company_size.lower() in ["small", "startup", "private"]:
                counterparty_risk_score += 0.15
                evidence.append(f"Company size risk: {company_size}")

            # Relationship history
            relationship_duration = counterparty.get("relationship_years", 0)
            if relationship_duration < 1:
                counterparty_risk_score += 0.2
                evidence.append("New counterparty relationship")
            elif relationship_duration > 5:
                counterparty_risk_score -= 0.1  # Reduce risk for long relationship
                evidence.append(
                    f"Established relationship: {relationship_duration} years"
                )

            # Financial strength indicators
            financial_strength = counterparty.get("financial_strength", "")
            if financial_strength.lower() in ["weak", "distressed"]:
                counterparty_risk_score += 0.4
                evidence.append(f"Financial strength concern: {financial_strength}")
            elif financial_strength.lower() in ["strong", "excellent"]:
                counterparty_risk_score -= 0.1
                evidence.append(f"Strong financial position: {financial_strength}")

            return RiskFactor(
                factor_id="counterparty_risk",
                name="Counterparty Risk",
                description="Risk from counterparty characteristics and stability",
                weight=0.15,
                value=min(1.0, max(0.0, counterparty_risk_score)),
                confidence=0.6,
                evidence=evidence,
                source="counterparty_assessment",
            )

        except Exception as e:
            logger.warning(f"Counterparty risk assessment failed: {e}")
            return None

    def _assess_operational_financial_risk(
        self, document_data: Dict[str, Any]
    ) -> Optional[RiskFactor]:
        """Assess operational financial risk"""
        try:
            operational_risk_score = 0.0
            evidence = []

            # Document complexity risk
            doc_complexity = document_data.get("complexity_score", 0.5)
            if doc_complexity > 0.8:
                operational_risk_score += 0.3
                evidence.append(f"High document complexity: {doc_complexity:.2f}")
            elif doc_complexity > 0.6:
                operational_risk_score += 0.1
                evidence.append(f"Moderate complexity: {doc_complexity:.2f}")

            # Processing accuracy risk
            accuracy = document_data.get("processing_accuracy", 1.0)
            if accuracy < 0.9:
                operational_risk_score += 0.2
                evidence.append(f"Processing accuracy concern: {accuracy:.1%}")

            # Data completeness risk
            completeness = document_data.get("data_completeness", 1.0)
            if completeness < 0.8:
                operational_risk_score += 0.2
                evidence.append(f"Incomplete data risk: {completeness:.1%}")

            # Manual intervention requirement
            requires_manual = document_data.get("requires_manual_review", False)
            if requires_manual:
                operational_risk_score += 0.15
                evidence.append("Manual review required")

            # System integration risk
            integration_score = document_data.get("system_integration_score", 1.0)
            if integration_score < 0.7:
                operational_risk_score += 0.1
                evidence.append(f"Integration challenges: {integration_score:.2f}")

            if operational_risk_score > 0:
                return RiskFactor(
                    factor_id="operational_financial_risk",
                    name="Operational Financial Risk",
                    description="Risk from operational processes and data quality",
                    weight=0.1,
                    value=min(1.0, operational_risk_score),
                    confidence=0.8,
                    evidence=evidence,
                    source="operational_assessment",
                )

            return None

        except Exception as e:
            logger.warning(f"Operational financial risk assessment failed: {e}")
            return None

    def _calculate_financial_risk_score(
        self, risk_factors: List[RiskFactor], metrics: FinancialMetrics
    ) -> float:
        """Calculate overall financial risk score"""
        try:
            if not risk_factors:
                return 0.0

            # Weighted average of risk factors
            total_weight = sum(factor.weight for factor in risk_factors)
            if total_weight == 0:
                return 0.0

            weighted_score = (
                sum(factor.value * factor.weight for factor in risk_factors)
                / total_weight
            )

            # Adjustment for exposure size
            if metrics.total_value:
                if metrics.total_value > 50_000_000:  # Very large exposure
                    weighted_score *= 1.2
                elif metrics.total_value > 10_000_000:  # Large exposure
                    weighted_score *= 1.1
                elif metrics.total_value < 100_000:  # Small exposure
                    weighted_score *= 0.9

            return min(1.0, weighted_score)

        except Exception:
            return 0.5

    def _calculate_financial_confidence(
        self, risk_factors: List[RiskFactor], metrics: FinancialMetrics
    ) -> float:
        """Calculate confidence in financial risk assessment"""
        try:
            if not risk_factors:
                return 0.5

            # Average confidence weighted by factor importance
            total_weight = sum(factor.weight for factor in risk_factors)
            if total_weight == 0:
                return 0.5

            weighted_confidence = (
                sum(factor.confidence * factor.weight for factor in risk_factors)
                / total_weight
            )

            # Adjust for data completeness
            if metrics.total_value and metrics.currency:
                weighted_confidence = min(1.0, weighted_confidence * 1.1)
            elif not metrics.total_value:
                weighted_confidence *= (
                    0.8  # Lower confidence without key financial data
                )

            return weighted_confidence

        except Exception:
            return 0.5

    def _assess_financial_business_impact(
        self, risk_score: float, metrics: FinancialMetrics
    ) -> Dict[str, float]:
        """Assess business impact of financial risk"""
        try:
            base_impact = risk_score

            # Scale by financial exposure
            if metrics.total_value:
                if metrics.total_value > 10_000_000:
                    exposure_multiplier = 1.5
                elif metrics.total_value > 1_000_000:
                    exposure_multiplier = 1.2
                else:
                    exposure_multiplier = 1.0
            else:
                exposure_multiplier = 1.0

            return {
                "financial_impact": min(1.0, base_impact * exposure_multiplier),
                "operational_impact": min(1.0, base_impact * 0.8),
                "reputation_impact": min(1.0, base_impact * 0.6),
                "regulatory_impact": min(1.0, base_impact * 0.7),
            }

        except Exception:
            return {
                "financial_impact": 0.5,
                "operational_impact": 0.5,
                "reputation_impact": 0.5,
                "regulatory_impact": 0.5,
            }

    def _generate_financial_recommendations(
        self, risk_factors: List[RiskFactor], metrics: FinancialMetrics
    ) -> List[str]:
        """Generate financial risk mitigation recommendations"""
        recommendations = []

        try:
            for factor in risk_factors:
                if factor.value > 0.6:  # High risk factors
                    if "credit" in factor.name.lower():
                        recommendations.append("Implement enhanced credit monitoring")
                        recommendations.append(
                            "Consider credit insurance or guarantees"
                        )
                    elif "liquidity" in factor.name.lower():
                        recommendations.append("Improve cash flow forecasting")
                        recommendations.append("Establish backup financing facilities")
                    elif "market" in factor.name.lower():
                        recommendations.append("Consider hedging strategies")
                        recommendations.append("Monitor market conditions closely")
                    elif "currency" in factor.name.lower():
                        recommendations.append("Implement currency hedging")
                        recommendations.append("Consider natural hedging strategies")
                    elif "counterparty" in factor.name.lower():
                        recommendations.append("Enhanced due diligence required")
                        recommendations.append("Consider counterparty limits")

            # General recommendations based on exposure
            if metrics.total_value and metrics.total_value > 5_000_000:
                recommendations.append("Large exposure - board approval recommended")
                recommendations.append("Implement enhanced monitoring and reporting")

            return list(set(recommendations))  # Remove duplicates

        except Exception:
            return [
                "Implement comprehensive financial risk monitoring",
                "Regular risk assessment updates required",
            ]

    def _store_financial_history(
        self, document_id: str, metrics: FinancialMetrics, timestamp: datetime
    ):
        """Store financial history for trend analysis"""
        try:
            if document_id not in self.financial_history:
                self.financial_history[document_id] = []

            history_entry = (
                timestamp,
                {
                    "total_value": metrics.total_value,
                    "currency": metrics.currency,
                    "payment_terms_days": metrics.payment_terms_days,
                    "interest_rates": metrics.interest_rates,
                    "penalties": metrics.penalties,
                },
            )

            self.financial_history[document_id].append(history_entry)

            # Limit history to last 1000 entries
            if len(self.financial_history[document_id]) > 1000:
                self.financial_history[document_id] = self.financial_history[
                    document_id
                ][-1000:]

        except Exception as e:
            logger.warning(f"Failed to store financial history: {e}")

    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk_score >= self.risk_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif risk_score >= self.risk_thresholds["low"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _create_default_financial_assessment(
        self, document_data: Dict[str, Any]
    ) -> RiskAssessmentResult:
        """Create default assessment when processing fails"""
        return RiskAssessmentResult(
            assessment_id=str(uuid.uuid4()),
            document_id=document_data.get("document_id", "unknown"),
            risk_category=RiskCategory.FINANCIAL,
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.5,
            confidence_score=0.3,
            assessed_at=datetime.now(),
            risk_factors=[],
            business_impact={
                "financial_impact": 0.5,
                "operational_impact": 0.5,
                "reputation_impact": 0.5,
                "regulatory_impact": 0.5,
            },
            mitigation_recommendations=[
                "Manual financial risk assessment required",
                "Insufficient financial data for automated assessment",
            ],
            context={"error": "Financial assessment processing failed"},
        )
