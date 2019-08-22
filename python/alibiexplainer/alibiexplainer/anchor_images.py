from typing import Callable, List, Dict, Optional, Any
import kfserving
import logging
import joblib
import alibi
import numpy as np
from alibiexplainer.explainer_wrapper import ExplainerWrapper

logging.basicConfig(level=kfserving.server.KFSERVER_LOGLEVEL)


class AnchorImages(ExplainerWrapper):

    def __init__(self, predict_fn: Callable, explainer=Optional[alibi.explainers.AnchorImage], **kwargs):
        self.predict_fn = predict_fn
        self.anchors_image: Optional[alibi.explainers.AnchorImage] = explainer
        if self.anchors_image is None:
            self.prepare(**kwargs)
        else:
            # pass
            # self.prepare(**kwargs)
            self.input_shape = self.anchors_image.image_shape

    def prepare(self, training_data_url=None, input_shape_url=None, **kwargs):
        print(training_data_url)
        print(input_shape_url)
        if not training_data_url is None:
            logging.info("Loading training file %s" % training_data_url)
            training_data_file = kfserving.Storage.download(training_data_url)
            training_data = joblib.load(training_data_file)
        else:
            pass

        if not input_shape_url is None:
            logging.info("Loading input shape file %s" % input_shape_url)
            input_shape_file = kfserving.Storage.download(input_shape_url)
            self.input_shape = joblib.load(input_shape_file)
        else:
            raise Exception("Anchor_image requires input shape")

        logging.info("Creating AnchorImages")
        self.anchors_image = alibi.explainers.AnchorImage(predict_fn=self.predict_fn,
                                                           image_shape=self.input_shape)

    def explain(self, inputs: List) -> Dict:
        if not self.anchors_image is None:
            arr = np.array(inputs)
            # set anchor_images predict function so it always returns predicted class
            # See anchor_images.__init__
            if np.argmax(self.predict_fn(arr).shape) == 0:
                self.anchors_image.predict_fn = self.predict_fn
            else:
                self.anchors_image.predict_fn = lambda x: np.argmax(self.predict_fn(x), axis=1)
            anchor_exp = self.anchors_image.explain(arr)
            return anchor_exp
        else:
            raise Exception("Explainer not initialized")
