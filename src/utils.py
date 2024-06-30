from src.model import *  # Ensure ESM2Frozen is imported here
from src.data import DataloaderHandler
import pickle
from transformers import T5EncoderModel, T5Tokenizer, logging
import os

# Define the ModelAttributes class to handle different model configurations
class ModelAttributes:
    def __init__(self,
                 model_type: str,
                 class_type: pl.LightningModule,
                 alphabet,
                 embedding_file: str,
                 save_path: str,
                 outputs_save_path: str,
                 clip_len: int,
                 embed_len: int) -> None:
        self.model_type = model_type
        self.class_type = class_type
        self.alphabet = alphabet
        self.embedding_file = embedding_file
        self.save_path = save_path
        if not os.path.exists(f"{self.save_path}"):
            os.makedirs(f"{self.save_path}")
        self.ss_save_path = os.path.join(self.save_path, "signaltype")
        if not os.path.exists(f"{self.ss_save_path}"):
            os.makedirs(f"{self.ss_save_path}")

        self.outputs_save_path = outputs_save_path

        if not os.path.exists(f"{outputs_save_path}"):
            os.makedirs(f"{outputs_save_path}")
        self.clip_len = clip_len
        self.embed_len = embed_len

# Update get_train_model_attributes to include the new ESM-2 model
def get_train_model_attributes(model_type):
    if model_type == FAST:
        # Use ESM-2 instead of ESM-1b
        with open("models/ESM2_alphabet.pkl", "rb") as f:  # Updated path and file name
            alphabet = pickle.load(f)
        return ModelAttributes(
            model_type,
            ESM2Frozen,  # Updated to use ESM2Frozen model class
            alphabet,
            EMBEDDINGS[FAST]["embeds"],
            "models/models_esm2",  # Updated save path
            "outputs/esm2/",  # Updated outputs path
            1022,
            1280
        )
    elif model_type == ACCURATE:
        alphabet = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        return ModelAttributes(
            model_type,
            ProtT5Frozen,
            alphabet,
            EMBEDDINGS[ACCURATE]["embeds"],
            "models/models_prott5",
            "outputs/prott5/",
            4000,
            1024
        )
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_type)
