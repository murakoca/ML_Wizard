import json
import os

class Settings:
    def __init__(self, filepath="pipeline_config.json"):
        self.filepath = filepath
        self.data = {}
        self.load()

    def update(self, stage, key, value):
        if stage not in self.data:
            self.data[stage] = {}
        self.data[stage][key] = value
        self.save()

    def get(self, stage, key, default=None):
        return self.data.get(stage, {}).get(key, default)

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)