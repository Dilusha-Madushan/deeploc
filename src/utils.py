from src.model import *  # Ensure ESM2Frozen is imported here
from src.data import DataloaderHandler
import pickle
from transformers import T5Tokenizer, AutoTokenizer, AutoModelForMaskedLM, logging
import os

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

def get_train_model_attributes(model_type):
    if model_type == FAST:
        # Use the transformers library to load ESM-2
        model_name = "facebook/esm2_t12_35M_UR50D"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model and tokenizer: {e}")
            raise e

        return ModelAttributes(
            model_type,
            ESM2Frozen,  # Updated to use ESM2Frozen model class
            tokenizer,  # Using tokenizer instead of alphabet
            EMBEDDINGS[FAST]["embeds"],
            "models/models_esm2",  # Updated save path
            "outputs/esm2/",  # Updated outputs path
            1022,
            1280
        )
    elif model_type == ACCURATE:
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        return ModelAttributes(
            model_type,
            ProtT5Frozen,
            tokenizer,  # Using tokenizer instead of alphabet
            EMBEDDINGS[ACCURATE]["embeds"],
            "models/models_prott5",
            "outputs/prott5/",
            4000,
            1024
        )
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_type)
