# Trading Setup

## Read this first

Robinhood has **two very different realities** depending on what you want to trade:

| What you want to trade | Official API? | Verdict |
|---|---|---|
| **Crypto** (BTC, ETH, DOGE...) | **Yes** — Robinhood Crypto Trading API, launched May 2024 | Supported. This is what the code here targets. |
| **Stocks / ETFs / options** | **No** | There is no public, documented, supported Robinhood API for equities. |

For stocks there is a popular package, `robin_stocks`, that drives Robinhood's
*private* mobile app API. It is not sanctioned. Robinhood's Terms of Service
prohibit accessing the service by automated means, and accounts using it have
been rate-limited and locked. Your money and your positions sit behind that
account. **This repo does not use it.**

If you want to automate *equities*, use a broker with a real API and real
paper trading — Alpaca, Tradier, or Interactive Brokers. See
`docs/equities-alternatives.md`.

## What's here

```
trading/
  rh_crypto.py        Signed client for the official Robinhood Crypto API
  config.py           Loads credentials + safety rails from .env
  generate_keys.py    Creates your Ed25519 keypair
  check_setup.py      Read-only smoke test — run this first
  strategy_example.py Worked example, dry-run by default
  test_signing.py     Unit tests for request signing (no network, no creds)
```

## Setup

```bash
cd trading
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 1. Generate a keypair

```bash
python generate_keys.py
```

This prints a **public key** and a **private key**, both base64. The private
key never leaves your machine.

### 2. Enroll the public key with Robinhood

Log in to Robinhood on **web classic**, go to your crypto account's API
settings, create a new API credential, and paste in the **public key**.
Robinhood gives you back an **API key** (a UUID). Note the permissions and
any IP allowlist you set there — an allowlist is the single most effective
control you have.

### 3. Fill in .env

```bash
cp .env.example .env
$EDITOR .env
```

`.env` is gitignored. Never commit it. If a key is ever pasted into a chat,
a log, or a commit, revoke it in Robinhood and generate a new one — treat it
as burned.

### 4. Verify, read-only

```bash
python check_setup.py
```

This only reads: account, holdings, and a quote. It places no orders. If
this passes, your signing and enrollment are correct.

## Safety rails

`RH_DRY_RUN=true` is the default and is enforced in code — `place_order()`
raises unless you explicitly set it false. `RH_MAX_ORDER_USD` caps the
notional value of any single order. Both exist because the failure mode of a
trading bug is not a stack trace, it is a filled order.

Work in this order: read-only → dry run → one manual live order at minimum
size → automation. Do not skip a step.

## Caveats

Robinhood's docs (docs.robinhood.com/crypto/trading/) were not reachable from
this sandbox, so the endpoint paths and the order-body shape in `rh_crypto.py`
were written from the published signing scheme and community clients.
**Diff them against the official docs before you trade live.** The signing
logic itself is unit-tested and is the part most likely to bite you.

## This is not financial advice

Nothing here is a recommendation to trade or a strategy that makes money.
It is plumbing. Crypto trades 24/7, is not SIPC-insured, and an automated
loop can lose money while you sleep in a way manual trading cannot.
