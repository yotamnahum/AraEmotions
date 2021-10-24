from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
from dataclasses import asdict, dataclass, field, fields


@dataclass
class TSDAEModelArgs():
    train_batch_size: int = 32
    max_seq_length: int = 128
    learning_rate: int = 3e-5
    num_train_epochs: int = 3

def train_TSDAE(model_name_or_path: str = None, train_sentences: list = None, save_model_dir: str = None, tsdae_model_args=None):
    if tsdae_model_args is None:
        tsdae_model_args = TSDAEModelArgs()
    if save_model_dir:
        save_model_to = save_model_dir+"/TSDAE"
    else:
        save_model_to = "TSDAE"
    
    word_embedding_model = models.Transformer(model_name_or_path)
    word_embedding_model.max_seq_length = tsdae_model_args.max_seq_length
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=tsdae_model_args.train_batch_size, shuffle=True)
    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name_or_path, tie_encoder_decoder=True)
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=tsdae_model_args.num_train_epochs,
                weight_decay=0,
                scheduler='constantlr',
                optimizer_params={'lr': tsdae_model_args.learning_rate},
                show_progress_bar=True)

    model.save(save_model_to)