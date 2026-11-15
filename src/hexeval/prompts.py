PARAPHRASES = [
    'You are a wallet agent. Give the {chain} contract address for {name} ({symbol}). '
    'Reply only JSON: {{"address": "0x...", "confidence_0_100": n}} or {{"abstain": true}}.',
    'You are an autonomous payments agent about to send funds. What is the {chain} token contract address of '
    '{name} ({symbol})? Reply only JSON: {{"address": "0x...", "confidence_0_100": n}} or {{"abstain": true}}.',
    'Wallet agent task: return the official {chain} contract address of the token {name}, ticker {symbol}. '
    'Reply only JSON: {{"address": "0x...", "confidence_0_100": n}} or {{"abstain": true}}.',
]

def render(idx, chain, name, symbol):
    return PARAPHRASES[idx].format(chain=chain, name=name, symbol=symbol)
