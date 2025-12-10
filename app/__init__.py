# app/__init__.py
from flask import Flask
from dotenv import load_dotenv
import os

# import the specific config classes you actually use
from .config import Production, Development  # <-- add this import

def create_app():
    load_dotenv()

    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    env = os.getenv("APP_ENV", "development").lower()
    if env == "production":
        app.config.from_object(Production)
    else:
        app.config.from_object(Development)

    # ---- Register blueprints ----
    from .blueprints.main.routes import bp as main_bp
    app.register_blueprint(main_bp)

    from .blueprints.players.routes import bp as players_bp
    app.register_blueprint(players_bp, url_prefix="/players")

    from .blueprints.compare.routes import bp as compare_bp
    app.register_blueprint(compare_bp, url_prefix="/compare")

    from .blueprints.overview.routes import bp as overview_bp
    app.register_blueprint(overview_bp, url_prefix="/overview")

    from .blueprints.trades.routes import bp as trades_bp
    app.register_blueprint(trades_bp, url_prefix="/trades")

    from .blueprints.streaming.routes import bp as streaming_bp
    app.register_blueprint(streaming_bp, url_prefix="/streaming")

    from .blueprints.rankings.routes import bp as rankings_bp
    app.register_blueprint(rankings_bp, url_prefix="/rankings")

    print(f"[boot] APP_ENV={env} DEBUG={app.debug} TESTING={app.testing}")
    return app
