from typing import Callable, List, Dict, Optional, Any
import kfserving
import logging
import os
import joblib
import alibi
from alibiexplainer.explainer_method import ExplainerMethodImpl
import numpy as np
import pandas as pd

logging.basicConfig(level=kfserving.server.KFSERVER_LOGLEVEL)


class AnchorImages(ExplainerMethodImpl):

    def __init__(self, predict_fn: Callable):
        self.predict_fn = predict_fn
        self.image_shape = None
        self.anchors_images: Optional[alibi.explainers.AnchorImage] = None

    def validate(self, training_data_url: Optional[str]):
        if training_data_url is not None:
            training_data_file = kfserving.Storage.download(training_data_url)
            training_data = joblib.load(training_data_file)
            self.image_shape = training_data.shape[1:]
            self.image_shape = self.image_shape.reshape(1, self.image_shape)
        else:
            pass

    def prepare(self, training_data_url: str):

        image_shape_str = os.environ.get("IMAGE_SHAPE_STRSCV")
        if not image_shape_str is None:
            logging.info("Image shape: %s" % image_shape_str)
            self.image_shape = tuple(image_shape_str.split(","))

        if self.image_shape is not None:
            self.anchors_images = alibi.explainers.AnchorImage(predict_fn=self.predict_fn,
                                                               image_shape=self.image_shape)
        else:
            raise Exception("Anchor images requires image shape")

    def explain(self, inputs: List) -> Dict:
        if not self.anchors_images is None:
            arr = np.array(inputs)
            anchor_exp = self.anchors_images.explain(arr)
            return anchor_exp
        else:
            raise Exception("Explainer not initialized")
