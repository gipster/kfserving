from typing import Callable, List, Dict, Optional, Any
import kfserving
import logging
import joblib
import alibi
import numpy as np
import pandas as pd
from alibiexplainer.explainer_wrapper import ExplainerWrapper

logging.basicConfig(level=kfserving.server.KFSERVER_LOGLEVEL)


class AnchorImages(ExplainerWrapper):

    def __init__(self, predict_fn: Callable, explainer=Optional[alibi.explainers.AnchorImage], **kwargs):
        self.predict_fn = predict_fn
        self.anchors_images: Optional[alibi.explainers.AnchorImage] = explainer
        self.prepare(**kwargs)

    def prepare(self, training_data_url=None, input_shape_url=None, **kwargs):
        if not training_data_url is None:
            logging.info("Loading training file %s" % training_data_url)
            training_data_file = kfserving.Storage.download(training_data_url)
            training_data = joblib.load(training_data_file)
        else:
            raise Exception("Anchor_images requires training data")

        if not input_shape_url is None:
            logging.info("Loading input shape file %s" % input_shape_url)
            input_shape_file = kfserving.Storage.download(input_shape_url)
            input_shape = joblib.load(input_shape_file)
        else:
            raise Exception("Anchor_images requires feature names")

        logging.info("Creating AnchorImages")
        self.anchors_tabular = alibi.explainers.AnchorImage(predict_fn=self.predict_fn,
                                                            image_shape=input_shape)
        logging.info("Fitting AnchorImage")
        self.anchors_images.fit(training_data)

    def explain(self, inputs: List) -> Dict:
        if not self.anchors_images is None:
            arr = np.array(inputs)
            # set anchor_images predict function so it always returns predicted class
            # See anchor_images.__init__
            if np.argmax(self.predict_fn(arr).shape) == 0:
                self.anchors_images.predict_fn = self.predict_fn
            else:
                self.anchors_images.predict_fn = lambda x: np.argmax(self.predict_fn(x), axis=1)
            anchor_exp = self.anchors_images.explain(arr)
            return anchor_exp
        else:
            raise Exception("Explainer not initialized")
