import logging

import numpy as np
import umap.umap_ as umap
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

np.random.seed(11)


def load_st(name: str, device='cpu'):
    """
    Objective: load sentence transformer

    Inputs:
            - name, str: the name of the sentence transformer model
    - device, str: define the pytorch device in which the model will be loaded
    Outputs:
            - model, sentence_transformers
            .SentenceTransformer.SentenceTransformer:
            the model of sentence encoder
    """

    assert name in ['stsb-xlm-r-multilingual',
                    'distiluse-base-multilingual-cased-v2',
                    'distiluse-base-multilingual-cased-v1',
                    'quora-distilbert-multilingual',
                    'paraphrase-xlm-r-multilingual-v1',
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    'paraphrase-multilingual-mpnet-base-v2',
                    'LaBSE'], 'ValueError: {} is not a valid model name'.format(name)

    model = SentenceTransformer(name, device=_device(device))

    logging.info('{} model loaded'.format(name))

    return model


def _device(device: Optional[str] = None) -> str:
    """Handle the device that we want to use to a proper device"""

    device = device if device else "cpu"
    logger.debug("Trying to set {} as device".format(device))

    if device not in ["cuda", "mps", "cpu"]:
        logger.debug("Device {} is setted to be used".format(device))
        return "cpu"

    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    cuda_available = torch.cuda.is_available()

    if "mps" == device and not mps_available:
        logger.debug("Device mps is not available")
        device = "cpu"
    elif "cuda" == device and not cuda_available:
        logger.debug("Device cuda is not available")
        device = "cpu"
    elif device is None:
        device = "cpu"

    logger.debug("Device {} is setted to be used".format(device))

    return device


def get_embeddings(
        texts: List[str],
        model: SentenceTransformer,
        show_progress_bar: bool = False):
    """
    Objective: get the embeddings of the texts

    Inputs:
            - texts, list/arrays: the texts to encode
            - model, sentence_transformers.SentenceTransformer
            .SentenceTransformer: the model of sentence encoder
    - show_progress_bar, bool: to show or not the progress bar
    Outputs:
            - embeddings, array: the encoded texts
    """

    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)

    return embeddings


def reduce_pipe(embeddings, y: Optional[Any] = None, **kwargs):
    """
    Objective: From the existing embeddings
    reduce the dimension of the data through a pipeline:
        1. Standard scaler
        2. Truncated SVD (equivalent to PCA as we standardize the data)
        3. UMAP algorithm with cosine metric, n_components to be highher

    Inptus:
        - embeddings, np.array: the embeddings
        - y, np.array or False: if the clusters already exists to get
                supervised UMAP
        - **kwargs, dict: the arguments as keywords for:
            - svd
            (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
            - umap
            (https://github.com/lmcinnes/umap)
    Outputs:
            - reduce_pipe, sklearn.pipeline.Pipeline: the pipeline
                model to reduce the embeddings
            - embeddings_reduced, np.array the embedding with reduced dimension
    """

    reduce_pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('SVD', TruncatedSVD(
                n_components=kwargs.get('n_component_pca', 50))),
            ('UMAP', umap.UMAP(
                n_neighbors=kwargs.get('n_neighbors', 50),
                metric='cosine',
                n_components=kwargs.get('n_components_umap', 10),
                init='spectral',
                transform_seed=42,
                random_state=11))
        ]
    )

    reduce_pipe.fit(embeddings, y=y) if y else reduce_pipe.fit(embeddings)
    embeddings_reduced = reduce_pipe.transform(embeddings)

    return reduce_pipe, embeddings_reduced
