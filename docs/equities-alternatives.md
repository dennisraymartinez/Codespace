# Automating stocks and options: the honest options

Robinhood publishes no supported API for equities. If your goal is to
automate **stocks, ETFs or options**, you need a different broker. This is
not a workaround — it is the only path that does not put your account at
risk.

## Why not `robin_stocks`

`robin_stocks` is a well-written package, and it works. It works by driving
Robinhood's *private* mobile-app API — the same endpoints the phone app
uses, reverse-engineered. What that means in practice:

- **It violates Robinhood's Terms of Service.** Their ToS prohibits
  accessing the service through automated means or scripts.
- **There is no contract.** Robinhood can change or break those endpoints
  without notice, mid-strategy, with an open position.
- **It needs your actual login.** Username, password, and MFA — full account
  access, not a scoped API key you can revoke.
- **Accounts do get restricted.** Rate-limiting and lockouts are documented
  by users, and a locked account during a drawdown is its own problem.

For paper trading or reading your own data it is a calculated risk. For a
loop placing real orders against real money, the risk is badly priced.

## Brokers with a real API

| Broker | Paper trading | Notes |
|---|---|---|
| **Alpaca** | Yes, free, unlimited | Commission-free US equities, REST + websockets, good Python SDK. The usual starting point. |
| **Tradier** | Yes (sandbox) | Equities and options, straightforward REST, small monthly fee for market data. |
| **Interactive Brokers** | Yes | The most capable and the most complex. Global markets. The API is a real project to learn. |

## Suggested path

1. **Alpaca paper account.** Free, no money at risk, an official API, and a
   near-identical interface when you switch to live. Prove your strategy
   here first.
2. **Keep the same guards this repo uses** — dry run by default, a hard
   per-order notional cap, credentials in `.env` and never in source.
3. **Go live small**, if at all, and only after the paper results survive a
   period you did not use to build the strategy.

## The part that matters more than the plumbing

Getting orders to route is the easy half, and it is the half this repo
solves. The hard half is having an edge. Most automated retail strategies
lose money to fees, spread, and overfitting to the backtest window. Build
the plumbing carefully, then be skeptical of your own signal.
