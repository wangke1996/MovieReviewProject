from backend import creat_app
from backend.sentiment import *
import argparse

app = creat_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5002)
    parser.add_argument('--debug', default=False, action="store_true")
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
