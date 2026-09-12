# Robinhood Crypto trading bot — setup

Read-only scaffolding plus the signed API client. No order-placing logic is
wired up yet; `check_setup.py` only issues GET requests.

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
```

## Files

| File | Purpose |
| --- | --- |
| `generate_keys.py` | Generates an Ed25519 keypair. Prints only — writes nothing to disk. |
| `robinhood_client.py` | Signs and sends requests. Read helpers only so far. |
| `check_setup.py` | Preflight: credentials, signing, enrollment, market data, safety rails. Exit 0 = all clear. |
| `.env.example` | Template for credentials and the safety rails. |

## Safety rails in `.env`

- `TRADING_ENABLED` — master kill switch, `false` by default. Leave it off
  until you have watched a strategy run.
- `MAX_ORDER_USD` — hard cap on any single order's notional value.
- `ALLOWED_SYMBOLS` — allow-list of pairs the bot may touch.

## Secrets

`.env` and any `*.pem` / `*.key` are gitignored. The private key is a bearer
credential: whoever holds it plus your API key can act on your account within
the permissions you granted. It is never written to disk by
`generate_keys.py` — copy it straight from the terminal into `.env`.

Grant the API credential read-only permission first. Add trading permission
only when you are ready to place real orders.
