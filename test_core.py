from datetime import date

import pytest

from engine import Bundle, Page, Policy, Purchase, Rule, assess, verify


POLICY_TEXT = "Items may be returned within 30 days of delivery."
POLICY_URL = "https://shop.example.com/returns"
DELIVERY_DATE = date(2026, 10, 5)


def verified_policy(quote=POLICY_TEXT):
    page = Page(id="returns", url=POLICY_URL, text=POLICY_TEXT)
    rule = Rule(
        id="return-window",
        kind="window",
        purpose="eligibility",
        source=page.id,
        quote=quote,
        time_quote=quote,
        days=30,
        basis="calendar",
        start="receipt_date",
        boundary="within",
    )
    policy = verify(Policy(rules=[rule], complete=True), [page])
    return Bundle(policy=policy, pages=[page])


@pytest.mark.parametrize(
    ("return_day", "status"),
    [
        (3, "Conditionally eligible"),
        (4, "Needs information"),
        (5, "Outside stated window"),
    ],
)
def test_return_window_starts_at_delivery(return_day, status):
    purchase = Purchase(purchase_date=date(2026, 10, 1), receipt_date=DELIVERY_DATE)
    result = assess(purchase, verified_policy(), on=date(2026, 11, return_day))
    assert result.status == status
    assert result.deadline == date(2026, 11, 4)


def test_missing_delivery_date_prompts_follow_up():
    purchase = Purchase(purchase_date=date(2026, 10, 1))
    result = assess(purchase, verified_policy(), on=date(2026, 10, 20))
    assert result.status == "Needs information"
    assert result.missing == "receipt_date"


def test_unsupported_policy_quote_cannot_produce_approval():
    purchase = Purchase(receipt_date=DELIVERY_DATE)
    bundle = verified_policy("Items may be returned within 60 days of delivery.")
    assert not bundle.policy.rules
    assert assess(purchase, bundle, on=date(2026, 10, 20)).status == "Unknown"