# Forex Trading Bot

Forex trading bot that executes trades automatically.

> ⚠️ Disclaimer: This project is for educational purposes only. Automated trading carries significant financial risk. Use at your own risk and test thoroughly with paper trading before using real money.

## Features
- Connects to a broker/API to fetch market data and place orders
- Strategy modules (simple moving average, RSI, etc.)
- Risk management (position sizing, stop loss, take profit)
- Logging and basic performance tracking

## Requirements
- Python 3.8+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

If there's no requirements.txt, install common packages used for trading bots (example):

```bash
pip install pandas numpy ta requests websocket-client
```

## Configuration
1. Create a copy of the example config (if provided): `cp config.example.json config.json`
2. Fill in your API keys and broker/account details in `config.json` or set them as environment variables.

Common environment variables:
- BROKER_API_KEY
- BROKER_API_SECRET
- ACCOUNT_ID

Keep your credentials secure. Do not commit API keys to source control.

## Usage
Run the bot from the project root:

```bash
python main.py
```

Adjust the strategy and configuration files as needed. If the repo uses a different entry point (e.g., `bot.py` or a package), run that file instead.

## Testing and Paper Trading
Always test strategies using historical/backtesting tools or a broker's paper trading/sandbox environment before trading with real funds.

## Contributing
Contributions are welcome. If you add new strategies or broker integrations, include tests or examples and update the README.

## License
Specify a license for this project (e.g., MIT). If you don't have one yet, add a `LICENSE` file.

## Contact
If you have questions, open an issue or contact the repository owner.
