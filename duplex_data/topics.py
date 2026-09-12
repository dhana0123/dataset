"""India scenario / topic bank for synthetic duplex dialogues."""

from __future__ import annotations

import random
from dataclasses import dataclass

DOMAINS: dict[str, list[str]] = {
    "recruitment": [
        "HR phone screen for a junior software engineer role",
        "Recruiter calling about a backend internship in Hyderabad",
        "Follow-up interview scheduling for a data analyst position",
        "Candidate asking about salary band for a customer success role",
        "HR verifying notice period and joining date",
    ],
    "customer_support": [
        "Banking app OTP not received; agent helps reset",
        "Prepaid recharge failed but money deducted",
        "Order delayed; customer wants refund or replacement",
        "Broadband outage complaint and ticket creation",
        "Wrong item delivered; arrange return pickup",
    ],
    "banking": [
        "Unauthorized UPI debit dispute",
        "Credit card limit increase request",
        "New savings account KYC clarification",
        "EMI foreclosure charges explanation",
        "Blocked debit card due to wrong PIN",
    ],
    "telecom": [
        "SIM activation delay after port-in",
        "High unexpected data charges this month",
        "Postpaid plan upgrade recommendations",
        "Network coverage issue in a locality",
        "eSIM conversion help for a new phone",
    ],
    "ecommerce": [
        "Missing package from a recent order",
        "Return window closing; request pickup",
        "Coupon not applied at checkout",
        "Seller cancelled order; ask for refund timeline",
        "Gift order address change before dispatch",
    ],
    "healthcare": [
        "Book a general physician teleconsult",
        "Lab report delay follow-up",
        "Pharmacy delivery status for medicines",
        "Insurance cashless claim document checklist",
        "Reschedule a diagnostic appointment",
    ],
    "government_schemes": [
        "Rythu Bandhu / farmer scheme eligibility questions",
        "Aadhaar–bank linking help desk call",
        "Ration card update status enquiry",
        "Scholarship application document clarification",
        "Municipal property tax payment issue",
    ],
    "education": [
        "Online course fee refund request",
        "Exam centre change request",
        "Hostel allotment confirmation call",
        "Parent asking about fee installment options",
        "Certificate reprint request from college office",
    ],
    "travel": [
        "Train ticket cancellation and refund",
        "Flight delay rebooking options",
        "Hotel booking date change",
        "Cab driver not arriving; reassign trip",
        "Bus ticket seat preference change",
    ],
    "utilities": [
        "Electricity bill spike clarification",
        "Gas cylinder booking delay",
        "Water supply complaint in the area",
        "New electricity connection application status",
        "Meter reading dispute",
    ],
}

# agent = Moshi/assistant side, user = caller/customer
DOMAIN_ROLES: dict[str, tuple[str, str]] = {
    "recruitment": ("HR recruiter", "job candidate"),
    "customer_support": ("customer support agent", "customer"),
    "banking": ("bank support agent", "account holder"),
    "telecom": ("telecom support agent", "subscriber"),
    "ecommerce": ("ecommerce support agent", "shopper"),
    "healthcare": ("clinic helpdesk agent", "patient"),
    "government_schemes": ("helpline officer", "citizen"),
    "education": ("admin office staff", "student or parent"),
    "travel": ("travel support agent", "traveler"),
    "utilities": ("utility support agent", "resident"),
}

CITIES = [
    "Hyderabad",
    "Bengaluru",
    "Chennai",
    "Mumbai",
    "Pune",
    "Delhi",
    "Ahmedabad",
    "Jaipur",
    "Kochi",
    "Visakhapatnam",
    "Warangal",
    "Vijayawada",
]

NAMES = [
    "Ravi",
    "Sita",
    "Ananya",
    "Karthik",
    "Priya",
    "Arjun",
    "Lakshmi",
    "Vikram",
    "Meena",
    "Suresh",
    "Divya",
    "Rahul",
]


@dataclass(frozen=True)
class ScenarioDraw:
    domain: str
    scenario: str
    agent_role: str
    user_role: str
    city: str
    person_name: str


def list_domains() -> list[str]:
    return sorted(DOMAINS.keys())


def parse_domains(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return list_domains()
    out = []
    for part in str(raw).split(","):
        d = part.strip().lower().replace(" ", "_")
        if not d:
            continue
        if d not in DOMAINS:
            raise SystemExit(f"Unknown domain {d!r}. Choose from: {', '.join(list_domains())}")
        out.append(d)
    return out or list_domains()


def _name_city(rng: random.Random) -> tuple[str, str]:
    try:
        from faker import Faker

        fake = Faker("en_IN")
        fake.seed_instance(rng.randint(0, 10_000_000))
        return fake.first_name(), fake.city()
    except Exception:
        return rng.choice(NAMES), rng.choice(CITIES)


def sample_scenario(
    domains: list[str] | None = None,
    *,
    rng: random.Random | None = None,
) -> ScenarioDraw:
    rng = rng or random.Random()
    pool = domains or list_domains()
    domain = rng.choice(pool)
    scenario = rng.choice(DOMAINS[domain])
    agent_role, user_role = DOMAIN_ROLES[domain]
    name, city = _name_city(rng)
    # Ground the scenario in a place/person without rewriting the template heavily
    scenario_full = f"{scenario} (caller {name}, city {city})"
    return ScenarioDraw(
        domain=domain,
        scenario=scenario_full,
        agent_role=agent_role,
        user_role=user_role,
        city=city,
        person_name=name,
    )
