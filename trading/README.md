# Robinhood Crypto trading bot

Signed API client, order placement, and the safety rails that gate it.

## One-time setup

```bash
cd trading
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python generate_keys.py     # prints a keypair; enroll the PUBLIC key on
                            # robinhood.com (classic web) > Account >
                            # Crypto > API keys

cp .env.example .env        # paste the API key + the PRIVATE key
python check_setup.py       # reads only, places nothing
python test_safety.py       # rails test suite, no network
```

## Placing an order

Dry run is the default, and you have to ask twice for a live order:
`--execute` on the command line **and** `TRADING_ENABLED=true` in `.env`.
Either one alone sends nothing.

Size the order either in the asset (`--quantity`) or in dollars (`--usd`).

```bash
# show exactly what would be sent, touch nothing
python place_order.py --symbol BTC-USD --side buy --usd 50

# live $50 market buy
python place_order.py --symbol BTC-USD --side buy --usd 50 --execute

# live buy of a specific quantity
python place_order.py --symbol BTC-USD --side buy --quantity 0.0005 --execute

# live limit sell
python place_order.py --symbol ETH-USD --side sell --quantity 0.01 \
    --type limit --limit-price 3200 --execute
```

`--usd` is a **ceiling on spend, not a target**. The amount is divided by the
quote already padded with the slippage buffer, and the resulting quantity is
rounded *down* to the pair's increment — so `--usd 50` produces an order the
rails estimate at $50.00 or less, never $50.50. It is checked against the
exchange's own minimum and maximum order size too.

Sizing in dollars is not a way around the rails. `--usd 50` against
`MAX_ORDER_USD=25` is still refused.

Exit codes: `0` sent (or dry run completed), `1` refused by the rails,
`2` Robinhood rejected it, `3` connection failed.

## Files

| File | Purpose |
| --- | --- |
| `generate_keys.py` | Generates an Ed25519 keypair. Prints only — writes nothing to disk. |
| `robinhood_client.py` | Signs and sends requests. Raw transport, applies **no** rails. |
| `safety.py` | The rails, the order intent, and the daily usage ledger. |
| `trader.py` | The single choke point. Everything goes through `Trader.submit`. |
| `place_order.py` | CLI for one order. Dry run unless `--execute`. |
| `check_setup.py` | Preflight: rails, usage, credentials, signing, market data. Exit 0 = all clear. |
| `test_safety.py` | 29 tests over the rails and the choke point. Fakes the client, no network. |

`robinhood_client.py` can place an order without any rail — it is deliberately
dumb transport. Application code must go through `Trader`, which is where the
rails are enforced.

## The rails

Configured in `.env`, enforced in `safety.py` before the request body is
built, so a refused order never reaches the network. All violations are
reported at once rather than one per run.

| Rail | Stops |
| --- | --- |
| `TRADING_ENABLED` | Everything. Master kill switch, `false` by default. |
| `ALLOWED_SYMBOLS` | Trading a symbol you didn't mean to name. |
| `MAX_ORDER_USD` | One order being larger than intended. |
| `SLIPPAGE_BUFFER_PCT` | Slippage carrying a market fill past the cap — the cap is tested against the quote padded by this much. |
| `MAX_LIMIT_DEVIATION_PCT` | Fat fingers: a limit price far off the mid (misplaced decimal) is rejected instead of resting on the book. |
| `MAX_DAILY_USD` | A looping bug draining the account. |
| `MAX_ORDERS_PER_DAY` | The same, by count — catches a fast loop of individually tiny orders. |
| holdings check | Selling more than you hold. |
| quote required | Sizing an order blind when market data is unavailable. |

Buys are priced off the ask and sells off the bid — the side that would
actually fill — so the notional estimate is never optimistic.

The daily caps live in `state/ledger.json` (gitignored) and reset on the UTC
day boundary. Persisting them is the point: a crash-loop that reset the
counters in memory would make the daily caps meaningless.

Cancelling is never gated. You can always cancel an open order, including
while the kill switch is on.

## Two failure modes worth knowing

**A POST that never answers.** If the connection dies after the order is
sent, the order may exist. `Trader` charges it against the daily budget
(over-counting costs headroom; under-counting would let a flapping
connection place unlimited orders), logs it as
`unknown_may_have_been_placed`, and tells you the `client_order_id` to
reconcile. Retry with the same id via `--client-order-id` — Robinhood dedupes
on it, so a duplicate cannot fill.

**No quote.** Market data being unavailable is a refusal, not a default. The
rails will not size an order against a missing price.

Every attempt — refused, dry-run, placed, or unknown — appends to
`state/orders.jsonl` (gitignored).

## Secrets

`.env` and any `*.pem` / `*.key` are gitignored. The private key is a bearer
credential: whoever holds it plus your API key can act on your account within
the permissions you granted. It is never written to disk by
`generate_keys.py` — copy it straight from the terminal into `.env`.

Grant the API credential read-only permission first. `check_setup.py` needs
nothing more. Add trading permission only when you are ready to place real
orders.
