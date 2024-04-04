from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio


def load_separator():

    model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham-enhancement')

    return model

def separate_audio(model, path, save_path):
    est_sources = model.separate_file(path=path)
    torchaudio.save(save_path, est_sources[:, :, 0].detach().cpu(), 8000)
    return print("Separation done!")

