from flask import Flask
from .config import Config
from dotenv import load_dotenv

def create_app():
    # Load .env in all environments (safe if vars already set)
    load_dotenv()

    # Tell Flask to use the top-level folders you already have
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(Config)

    # ---- Register blueprints ----
    from .blueprints.main.routes import bp as main_bp
    app.register_blueprint(main_bp)  # no prefix (home, nav)

    from .blueprints.players.routes import bp as players_bp
    app.register_blueprint(players_bp, url_prefix="/players")

    from .blueprints.compare.routes import bp as compare_bp
    app.register_blueprint(compare_bp, url_prefix="/compare")

    from .blueprints.overview.routes import bp as overview_bp
    app.register_blueprint(overview_bp, url_prefix="/overview")

    return app
