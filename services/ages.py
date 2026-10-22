"""Age arithmetic for reference photos (pure functions, no database).

Inputs come from the register form: an optional date of birth or approximate current age for the
case, and an optional approximate date for each photo ("YYYY", "YYYY-MM" or "YYYY-MM-DD").
"""
from datetime import date, datetime

DAYS_PER_YEAR = 365.25


def parse_date(value):
    """Lenient parse of YYYY / YYYY-MM / YYYY-MM-DD into a date (first day of the period), or None."""
    if not value:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt, size in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(text[:size], fmt).date()
        except ValueError:
            continue
    return None


def years_between(earlier, later):
    return (later - earlier).days / DAYS_PER_YEAR


def current_age(dob=None, age_reported=None, age_reported_on=None, today=None):
    today = today or date.today()
    born = parse_date(dob)
    if born:
        return years_between(born, today)
    if age_reported is not None:
        reported = parse_date(age_reported_on) or today
        return float(age_reported) + years_between(reported, today)
    return None


def age_at_photo(photo_date, dob=None, age_reported=None, age_reported_on=None, today=None):
    """Age of the person in a photo taken on photo_date, if enough is known."""
    taken = parse_date(photo_date)
    if taken is None:
        return None
    born = parse_date(dob)
    if born:
        return max(0.0, years_between(born, taken))
    now_age = current_age(dob, age_reported, age_reported_on, today)
    if now_age is None:
        return None
    return max(0.0, now_age - years_between(taken, today or date.today()))


def photo_years_old(photo_date=None, age_at=None, now_age=None, today=None):
    """How old the photo is today (years), from its date or from age-at-photo vs current age."""
    taken = parse_date(photo_date)
    if taken is not None:
        return max(0.0, years_between(taken, today or date.today()))
    if age_at is not None and now_age is not None:
        return max(0.0, float(now_age) - float(age_at))
    return None
