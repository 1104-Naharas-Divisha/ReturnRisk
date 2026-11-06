from datetime import date, datetime, timedelta
import re
from typing import Literal
from pydantic import Field
from engine import Engine, Model, Purchase, ServiceError, assess, today, product_label, norm

class DateValue(Model):
    field: Literal['purchase_date', 'receipt_date', 'return_date']
    absolute: date | None = None
    anchor: Literal[
        'today', 'purchase_date', 'receipt_date', 'unknown'
        ] = 'unknown'
    offset_days: int = Field(default=0, ge=-36500, le=36500)

class PurchaseChanges(Purchase):
    country: str | None = None

class Intent(Model):
    kind: Literal['purchase', 'general', 'hypothetical', 'compare', 'actual', 'casual']
    details: PurchaseChanges = Field(
        default_factory=PurchaseChanges
    )
    clear_fields: list[str] = Field(
        default_factory=list, max_length=16
    )
    dates: list[DateValue] = Field(
        default_factory=list, max_length=3
    )
    same_conditions: bool = False
    new_purchase: bool = False
    retailers: list[str] = Field(
        default_factory=list, max_length=3
    )
    casual_reply: str = ''
QUESTIONS = {'retailer': 'Which retailer did you buy it from?', 'product': 'What product are you asking about?', 'purchase_date': 'What date did you purchase the item?', 'receipt_date': 'What date did you receive the item?', 'return_date': 'What date are you planning to return it?', 'channel': 'Was it purchased in-store or online?', 'seller': 'Who sold it: the retailer directly or a marketplace seller?', 'platform': 'Which shopping platform did you use?', 'condition': 'Is it unused, gently used, used, or damaged?', 'packaging': 'Is the packaging factory-sealed, opened, or missing?', 'proof': 'What proof of purchase can you provide?', 'size_inches': 'What is the screen size in inches?', 'country': 'In which country did you purchase it?'}
INTENT_PROMPT = """Read the latest message using the purchase state and pending question.
Only put explicit changes in details. Leave unknown facts null. Do not infer receipt from purchase,
seller from retailer, or usage from opened packaging. When the user explicitly says bought
directly from the retailer, seller is that retailer. Normalize store/in-store to channel='in-store'. Keep condition and packaging separate.
Use a short category path, most specific type first, such as televisions, consumer electronics.
Set same_conditions=true when the user explicitly corrects the product type for the same transaction.
Use hypothetical for 'what if', compare for retailer comparisons, actual to return to the actual
purchase, general for policy information, casual for small talk, and purchase for other updates.
A standalone question about a different product is a new_purchase: do not copy the previous retailer, dates or condition. A correction to the same transaction is not a new purchase.
A retailer correction is purchase, not hypothetical. Set new_purchase for a different transaction.
A brief reply can answer the pending field. Put ALL dates in dates, never in details. Use an absolute
date only with a clear year; otherwise use today/purchase_date/receipt_date and an integer day offset.
Use unknown anchor if ambiguous. Python will calculate dates. A policy's duration is not a proposed
return date. Set same_conditions only for explicit carryover; clear_fields lists retracted facts.
For comparisons, supply up to three retailer names. A casual reply must contain no policy claims."""

def fresh_state():
    return {
        'actual': Purchase(), 
        'hypothetical': None, 
        'active': 'actual', 
        'scenario': {'return_date': None}, 
        'pending': None, 
        'history': [], 
        'last_policy': None, 
        'last_general': False, 
        'date_problem': None
    }

def make_reply(text, **details):
    reply = {
        'text': text, 
        'sources': [], 
        'notes': [], 
        'comparison': [], 
        'diagnostics': []
    }
    reply.update(details)
    return reply

def parse_date(text, reference):
    value = text.strip().rstrip('.!')
    offsets = {
        'today': 0, 
        'yesterday': -1, 
        'tomorrow': 1, 
        'a week ago': -7, 
        'next week': 7
    }
    if value.casefold() in offsets:
        return reference + timedelta(days=offsets[value.casefold()])
    value = re.sub(
        '(?<=\\d)(st|nd|rd|th)\\b', 
        '', 
        value, 
        flags=re.I
    )
    formats = ('%Y-%m-%d', '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y', '%d %B %Y', '%d %b %Y')
    for format_string in formats:
        try:
            return datetime.strptime(value, format_string).date()
        except ValueError:
            continue
    return None

def resolve_dates(values, dates, reference):
    pending = list(dates) #shallow copy of dates list
    if len({item.field for item in pending}) != len(pending):
        raise ValueError('duplicate_date_update')
    while pending:
        changed = False
        waiting_fields = {item.field for item in pending}
        for item in pending[:]:
            if item.absolute is not None:
                result = item.absolute
            elif item.anchor == 'today':
                result = reference + timedelta(days=item.offset_days)
            elif item.anchor in waiting_fields or not isinstance(values.get(item.anchor), date):
                continue
            else:
                result = values[item.anchor] + timedelta(days=item.offset_days)
            values[item.field] = result
            pending.remove(item)
            changed = True
        if not changed:
            for item in pending:
                values[item.field] = None
            return pending[0].field
    return None

def update_purchase(state, intent, reference):
    target = state['active']
    changes = intent.details.model_dump(exclude_unset=True, exclude_none=True)
    changes = {key: value for key, value in changes.items() if value != []}
    if intent.kind == 'actual':
        target = 'actual'
    elif intent.kind in {'hypothetical', 'compare'}:
        base = state[target] or state['actual']
        different_retailer = bool(intent.retailers) or (changes.get('retailer') and changes['retailer'] != base.retailer)
        keep = base.model_dump()
        if different_retailer and (not intent.same_conditions):
            keep = {key: keep[key] for key in ('product', 'category', 'size_inches', 'country')}
        state['hypothetical'] = Purchase(**keep)
        target = 'hypothetical'
    elif intent.new_purchase or (
        changes.get('category') and state[target].category
        and product_label(changes['category'][0]) != product_label(state[target].category[0])
        and not intent.same_conditions
    ):
        target = 'actual'
        state['actual'] = Purchase()
        state['hypothetical'] = None
        state['last_policy'] = None
        state['last_general'] = False
        state['pending'] = None
    values = state[target].model_dump()
    if changes.get('retailer') and changes['retailer'] != values['retailer']:
        state['last_policy'] = None
    defaults = Purchase().model_dump()
    for field in intent.clear_fields:
        if field in defaults:
            values[field] = defaults[field]
    values.update(changes)
    state['date_problem'] = resolve_dates(values, intent.dates, reference)
    purchase = Purchase(**values)
    state[target] = purchase
    state['active'] = target
    state['scenario'] = {'return_date': purchase.return_date}
    return purchase

def format_reply(purchase, bundle, result):
    text = f'**{result.status}** — {result.reason}'
    if result.deadline:
        text += f' Calculated deadline: **{result.deadline:%B %d, %Y}**.'
    if result.missing:
        text += '\n\n' + QUESTIONS.get(result.missing, f'Please clarify {result.missing}.')
    pages = {page.id: page for page in bundle.pages}
    rules = [rule for rule in bundle.policy.rules if rule.id in result.rule_ids]
    if not rules:
        rules = bundle.policy.rules[:4]
    sources = []
    for rule in rules:
        if rule.source not in pages:
            continue
        source = {'url': pages[rule.source].url, 'quote': rule.quote}
        if source not in sources:
            sources.append(source)
    return make_reply(text, sources=sources, notes=result.notes, diagnostics=bundle.diagnostics, retailer=purchase.retailer, status=result.status, deadline=str(result.deadline or ''))

def evaluate_purchase(state, engine, purchase, reference, refresh=False, general=False):
    if not purchase.retailer or (not purchase.product and (not general)):
        field = 'retailer' if not purchase.retailer else 'product'
        state['pending'] = field
        return make_reply(QUESTIONS[field])
    bundle = engine.research(purchase, refresh=refresh)
    result = assess(purchase, bundle, on=reference, general=general)
    state['last_policy'] = bundle
    state['last_general'] = general
    state['pending'] = result.missing
    return format_reply(purchase, bundle, result)

def compare_retailers(state, engine, purchase, retailers, reference):
    reply = make_reply("Here is the comparison using the stated facts. Each retailer's conditions still apply.")
    general = not (purchase.purchase_date or purchase.receipt_date)
    for retailer in retailers:
        scenario = purchase.model_copy(update={'retailer': retailer})
        part = evaluate_purchase(state, engine, scenario, reference, general=general)
        reply['comparison'].append({'Retailer': retailer, 'Assessment': part.get('status', 'Needs information'), 'Details': part['text'], 'Deadline': part.get('deadline', '')})
        for field in ('sources', 'notes', 'diagnostics'):
            reply[field].extend(part[field])
        state['hypothetical'] = scenario
    state['pending'] = None
    return reply

def answer_message(message, state, engine, reference):
    if norm(message) in {'ok', 'okay', 'thanks', 'thank you', 'okay thanks', 'ok thanks', 'thanks a lot', 'thank you very much'}:
        return make_reply("You're welcome!")
    pending = state['pending']
    if pending in {'purchase_date', 'receipt_date', 'return_date'}:
        parsed = parse_date(message, reference)
        if parsed is not None:
            purchase = state[state['active']].model_copy(update={pending: parsed})
            state[state['active']] = purchase
            state['scenario'] = {'return_date': purchase.return_date}
            state['date_problem'] = None
            return evaluate_purchase(state, engine, purchase, reference, general=state['last_general'])
    hypothetical = state['hypothetical']
    intent = engine.llm.ask(INTENT_PROMPT, {'message': message, 'today': str(reference), 'active': state['active'], 'pending': pending, 'actual': state['actual'].model_dump(mode='json'), 'hypothetical': hypothetical.model_dump(mode='json') if hypothetical else None, 'history': state['history'][-10:]}, Intent)
    if intent.kind == 'casual':
        return make_reply(intent.casual_reply or "You're welcome.")
    purchase = update_purchase(state, intent, reference)
    if state['date_problem']:
        field = state['date_problem']
        state['pending'] = field
        return make_reply('I need a clear starting event or date. ' + QUESTIONS[field])
    if intent.kind == 'compare' and intent.retailers:
        return compare_retailers(state, engine, purchase, intent.retailers, reference)
    return evaluate_purchase(state, engine, purchase, reference, general=intent.kind == 'general')

def handle(message, state, engine: Engine, on=None):
    try:
        reply = answer_message(message, state, engine, on or today())
    except (ServiceError, ValueError, OverflowError) as error:
        category = str(error) if isinstance(error, ServiceError) else 'invalid_structured_output'
        reply = make_reply("I couldn't reliably interpret or research that request. Please retry; no return eligibility has been confirmed.", diagnostics=[{'category': category}])
    state['history'].extend([{'role': 'user', 'text': message}, {'role': 'assistant', **reply}])
    state['history'] = state['history'][-60:]
    return reply

def refresh_policy(state, engine: Engine, on=None):
    purchase = state[state['active']]
    reply = evaluate_purchase(state, engine, purchase, on or today(), refresh=True, general=state['last_general'])
    state['history'].append({'role': 'assistant', **reply})
    state['history'] = state['history'][-60:]
    return reply
