import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dotenv import load_dotenv
load_dotenv()

import sys
from src.fetcher import load_stock
from src.features import add_features
from src.model import train
from src.predictor import predict_next_day, evaluate

ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
command = sys.argv[2] if len(sys.argv) > 2 else "train"

if command == "train":
    df = load_stock(ticker)
    df = add_features(df)
    results = train(df, ticker)
    print(f"\nDone! Best loss: {min(results['history'].history['val_loss']):.6f}")

elif command == "predict":
    result = predict_next_day(ticker)
    print(f"\n{result['ticker']} — {result['last_date']}")
    print(f"Last close:      ${result['last_close']}")
    print(f"Predicted next:  ${result['predicted_next']}")
    print(f"Expected change: {result['change']:+.2f} ({result['change_pct']:+.2f}%)")

elif command == "evaluate":
    metrics = evaluate(ticker)
    print(f"\nTest set evaluation:")
    print(f"MAE:  ${metrics['mae']}   (avg dollar error)")
    print(f"RMSE: ${metrics['rmse']}  (penalizes big misses)")
    print(f"MAPE: {metrics['mape']}%  (avg % error)")