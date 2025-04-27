import os
import json

class ConfigManager:
    def __init__(self, settings_file="settings.json"):
        self.settings = {}
        try:
            with open(settings_file, "r") as f:
                self.settings = json.load(f)
            print(f"✅ Loaded settings from {settings_file}")
        except FileNotFoundError:
            print(f"⚠️ Settings file '{settings_file}' not found. Using system environment variables only.")

    def get(self, key: str, default=None):
        # Try to get from environment first
        value = os.getenv(key)
        if value is not None:
            return value
        # Otherwise, get from loaded settings
        return self.settings.get(key, default)

    def get_json(self, key: str, default=None):
        # Same as get, but automatically parse if the value is JSON string
        value = self.get(key, default)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value
