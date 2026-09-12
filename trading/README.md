# Robinhood Crypto trading bot

Signed API client, order placement, and the safety rails that gate it.

## One-time setup

Windows (PowerShell):

If `python` opens the Microsoft Store instead of running, use `py` (the
Windows Python launcher) for the venv step. Once the venv is active,
`python` and `pip` resolve to it and work normally. Python 3.9 or newer is
enough — the code is checked against 3.9 syntax.

```powershell
cd trading
python -m venv venv                 # or: py -m venv venv
.\venv\Scripts\Activate.ps1   # the leading .\ is REQUIRED: without it
                               # PowerShell reads venv\... as a module-
                               # qualified command and fails with
                               # "The module 'venv' could not be loaded"
                               # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt

python generate_keys.py        # prints a keypair; enroll the PUBLIC key at
                               # robinhood.com WEB CLASSIC > crypto account
                               # settings > Add key (see "Getting a key" below)

copy .env.example .env         # paste the API key + the PRIVATE key
python check_setup.py          # reads only, places nothing
python test_safety.py          # rails test suite, no network
```

macOS / Linux:

```bash
cd trading
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python generate_keys.py
cp .env.example .env
python check_setup.py
python test_safety.py
```

The venv must be active in every new terminal before `python place_order.py`
— otherwise you get `ModuleNotFoundError: nacl`.

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

Sizing in dollars is not a way around the rails. `--usd 100` against
`MAX_ORDER_USD=60` is still refused.

Exit codes: `0` sent (or dry run completed), `1` refused by the rails,
`2` Robinhood rejected it, `3` connection failed.

A market order comes back `open`, not `filled` — the response is an
acknowledgement, not an execution. Check what actually happened with
`python orders.py`, or `python orders.py --watch <id>` to poll until it
settles. Robinhood may also adjust the quantity slightly from what was
sent, so the filled amount is the one that counts.

## Files

| File | Purpose |
| --- | --- |
| `generate_keys.py` | Generates an Ed25519 keypair. Prints only — writes nothing to disk. |
| `robinhood_client.py` | Signs and sends requests. Raw transport, applies **no** rails. |
| `safety.py` | The rails, the order intent, and the daily usage ledger. |
| `trader.py` | The single choke point. Everything goes through `Trader.submit`. |
| `place_order.py` | CLI for one order. Dry run unless `--execute`. |
| `set_api_key.py` | Writes `RH_API_KEY` into `.env` safely. Refuses a private key. |
| `arm.py` | Turns the kill switch on (asks first) or off (immediately). |
| `orders.py` | Shows recent orders and how they filled. Read-only. |
| `envfile.py` | Reads and writes single `.env` values without touching the rest. |
| `check_setup.py` | Preflight: rails, usage, credentials, signing, market data. Exit 0 = all clear. |
| `test_safety.py` | 68 tests over the rails and the choke point. Fakes the client, no network. |

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

## Getting a key from Robinhood

Available to Robinhood Crypto customers in the US. Generate the keypair
*first* — the web form asks you to paste the public key.

1. `python generate_keys.py`. It prints the **public** key and writes the
   private key straight into `.env` — you never see or copy it.
2. Sign in at robinhood.com on **web classic** (a desktop browser — this
   cannot be done in the mobile app).
3. Go to your **crypto account settings**.
4. Select **Add key**.
5. Paste the **PUBLIC** key from step 1 and name the credential.
6. Select the **API actions** to enable. Read-only is enough for
   `check_setup.py`; placing orders needs the trading action.
7. Robinhood shows you the **API key**. Set it with:

   ```powershell
   python set_api_key.py
   ```

   It prompts, strips any brackets or quotes you paste around the value,
   refuses a private key pasted by mistake, and writes it to the right
   variable. This is the only value you have to copy by hand, and this is
   the only step where hand-editing `.env` is easy to get wrong.

Credentials can be modified, disabled, or deleted later from the same page.
If the private key is ever exposed, delete the credential there first — that
revokes it immediately — then enroll a fresh keypair.

Sources: Robinhood's crypto API support article and the launch announcement
(robinhood.com/us/en/support/articles/crypto-api).

## Which API actions to enable

Robinhood's "Allowed API actions" map to endpoints this project calls. Tick
these:

| Robinhood action | Needed by |
| --- | --- |
| Read crypto accounts | `check_setup.py` authentication (`GET /accounts/`) |
| Read crypto quotes | every order — `best_bid_ask` prices the notional cap |
| Read crypto products | `--usd` sizing — quantity increment, min/max order size |
| Read crypto holdings | the sell rail — refuses selling more than you hold |
| Read crypto orders | reconciling an order whose POST never answered |
| **Place crypto orders without fee tiers** | placing orders (see below) |

All five read actions are genuinely required, not optional: the rails refuse
an order they cannot price, cannot size against the exchange's own
constraints, or (for a sell) cannot check holdings for. A credential with
only the order-placing action fails `check_setup.py` at section 5.

**Which "place" action:** they select the API version, not just a fee
schedule. "Place crypto orders **with** fee tiers" is v2; "**without** fee
tiers" is v1. This project posts to `/api/v1/crypto/trading/orders/`, so it
needs the **without fee tiers** action. Ticking only the v2 action leaves
orders failing while every read succeeds.

The trade-off is real: only v2 orders count toward the 30-day volume that
sets your fee tier (0.03%–0.85%). v1 is what this code speaks today.
Enabling both actions costs nothing and leaves the door open.

## TLS certificate errors

Symptom — from `pip install`, or from `check_setup.py` once installed:

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Cause: antivirus or a corporate proxy on the machine is intercepting HTTPS
and re-signing it with a private root certificate. Windows trusts that root
(so browsers work), but Python does not use the Windows trust store — it
uses its own bundled CA list, which has never heard of it.

Fix — hand Python the certificates Windows already trusts:

```powershell
python -c "import ssl,pathlib;p=[ssl.DER_cert_to_PEM_cert(c) for s in ('ROOT','CA') for c,e,t in ssl.enum_certificates(s) if e=='x509_asn'];pathlib.Path('win-ca.pem').write_text(''.join(p));print(len(p),'certs')"

pip install --cert win-ca.pem -r requirements.txt
```

The same interception will break the bot's own calls to Robinhood, so point
`requests` at the bundle too. Per terminal:

```powershell
$env:REQUESTS_CA_BUNDLE = "$PWD\win-ca.pem"
```

Or permanently, for your user account:

```powershell
setx REQUESTS_CA_BUNDLE "$PWD\win-ca.pem"
```

`win-ca.pem` is gitignored — it is machine-specific, and committing one
machine's trust store would be misleading everywhere else.

The `cacert.pem` in the repository root does **not** help here: it is a stock
public-root bundle, equivalent to what Python already uses and already
failing. Only a bundle containing the intercepting root works.

Last resort, if the export cannot be made to work:

```powershell
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

That skips verification for those hosts rather than fixing trust. It is a
poor trade when the packages being fetched will go on to sign trading
requests, so prefer the bundle.

## Secrets

`.env` and any `*.pem` / `*.key` are gitignored. The private key is a bearer
credential: whoever holds it plus your API key can act on your account within
the permissions you granted.

`generate_keys.py` writes it directly into `.env` and does not display it, so
there is no copy step to get wrong — a key you never see cannot be pasted
into a chat window, a ticket, or a screenshot. `--print-private` shows it
instead, for the rare case you need it somewhere else.

**If a private key is exposed, what to do depends on timing.** Before the
public key is enrolled, the keypair authorizes nothing: throw it away and
generate another, no cleanup needed. After enrolment, delete that credential
in your crypto account settings *first* — that revokes it immediately — then
enroll a fresh keypair.

Grant the API credential read-only permission first. `check_setup.py` needs
nothing more. Add trading permission only when you are ready to place real
orders.
