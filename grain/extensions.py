from flask_debugtoolbar import DebugToolbarExtension
from flask_static_digest import FlaskStaticDigest
from config import settings

debug_toolbar = DebugToolbarExtension()
flask_static_digest = FlaskStaticDigest()

# load up the medical library
settings.get_medical()