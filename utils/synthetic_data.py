import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from PIL import Image, ImageDraw
import os

class SyntheticDataGenerator:
    @staticmethod
    def tabular_data(task="classification", n_samples=500, n_features=10, n_classes=2):
        if task == "classification":
            X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                       n_classes=n_classes, random_state=42)
        else:
            X, y = make_regression(n_samples=n_samples, n_features=n_features,
                                   noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y
        return df

    @staticmethod
    def sentiment_texts(n_samples=100):
        positive = ["This is great!", "I love it", "Absolutely wonderful",
                    "Fantastic experience", "So happy with this", "Amazing product"]
        negative = ["Terrible service", "I hate it", "Very disappointing",
                    "Worst ever", "Not good at all", "Awful experience"]
        texts, labels = [], []
        for _ in range(n_samples // 2):
            texts.append(np.random.choice(positive))
            labels.append(1)
            texts.append(np.random.choice(negative))
            labels.append(0)
        return pd.DataFrame({"text": texts, "label": labels})

    @staticmethod
    def create_random_images(n=10, size=(416, 416), output_dir="synthetic_images"):
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i in range(n):
            img = Image.new('RGB', size, (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)))
            draw = ImageDraw.Draw(img)
            for _ in range(np.random.randint(1, 5)):
                x1 = np.random.randint(0, size[0]-50)
                y1 = np.random.randint(0, size[1]-50)
                draw.rectangle([x1, y1, x1+np.random.randint(20,80), y1+np.random.randint(20,80)],
                               outline="white", width=2)
            path = os.path.join(output_dir, f"synthetic_{i}.png")
            img.save(path)
            paths.append(path)
        return paths