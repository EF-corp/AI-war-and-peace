#from main import LanguageModel, BatchFirstTransformerEncoder, GreedyGenr, make_positional_encoding, make_target_dependency_mask
import torch
import youtokentome as yttm
from torch import nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from kivy.app import App
from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.config import Config
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivymd.theming import ThemeManager

Config.set('kivy', 'keyboard_mode', 'systemanddock')
# Window.size = (480, 800)
Builder.load_string('''


<AI>:
    rows: 5
    padding: 10
    text1: text1
    text2: text2
    lbl: lbl
    AnchorLayout:
        size_hint: 1, .25


        MDTextField:
            id: text1
            font_size: '30sp'
            multiline: False
            input_type: 'text'
            #input_filter: 'str'
            hint_text: 'введите текст'

    AnchorLayout:
        size_hint: 1, .25
        MDTextField:
            id: text2
            font_size: '30sp'
            multiline: False
            input_type: 'number'
            input_filter: 'int'
            hint_text: 'введите количество символов(n-грамм)'

    BoxLayout:
        AnchorLayout:
            anchor_y: 'top'
            #anchor_x: 'center_x'
            MDLabel:                           
                text: 'AI answer:'
                font_size: '40sp'
                bold: True
        MDLabel:

            id: lbl
            text: ''
            font_size: '33sp'
            italic: True


    GridLayout:
        cols: 1
        spacing: 10
        padding: [0, 30, 0, 0]
        size_hint: .9, .25
        MDRaisedButton:
            size_hint: 1, .5
            text: 'AI run'
            font_size: '35sp'
            bold: True
            #radius: (30, 30, 30, 30)
            on_release: root.ai()
        
    MDLabel:
        size_hint: 1, .15
        text: "made by EF.corp"
        italic: True

                    ''')

BPE_MODEL_FILENAME = "C:\\Users\\rober\\Downloads\\AI war_and_peace\\war_and_peace_bpe.yttm"
tokenizer = yttm.BPE(BPE_MODEL_FILENAME)


class GreedyGenr:
    def __init__(self, model, tokenizer, device='cuda', eos_token_id=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = eos_token_id

    def __call__(self, seed_text, max_steps_n=50):
        seed_tokens = self.tokenizer.encode([seed_text])[0]

        for _ in range(max_steps_n):
            in_batch = torch.tensor(seed_tokens).unsqueeze(0).to(self.device)
            best_next_token = self.model(in_batch)[0, -1].argmax()
            if best_next_token == self.eos_token_id:
                break

            seed_tokens.append(best_next_token)

        return self.tokenizer.decode([seed_tokens])[0]


class BatchFirstTransformerEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.impl = nn.TransformerEncoder(*args, **kwargs)
        self.initialize_weights()

    def forward(self, src, *args, **kwargs):
        src = src.transpose(0, 1).contiguous()  # MaxInLen  x BatchSize x EmbSize
        result = self.impl(src, *args, **kwargs)  # TargetLen x BatchSize x EmbSize
        result = result.transpose(0, 1).contiguous()  # BatchSize x TargetLen x EmbSize
        return result

    def initialize_weights(self):
        for param in self.impl.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, backbone, emb_dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.backbone = backbone
        self.out = nn.Linear(embedding_size, vocab_size)

    def forward(self, seed_token_ids):
        """
            seed_token_ids - BatchSize x MaxInLen
        """
        batch_size, max_in_length = seed_token_ids.shape

        seed_padding_mask = seed_token_ids == 0
        dependency_mask = make_target_dependency_mask(max_in_length) \
            .to(seed_token_ids.device)

        seed_embs = self.embeddings(seed_token_ids)  # BatchSize x MaxInLen x EmbSize
        pos_codes = make_positional_encoding(max_in_length,
                                             self.embedding_size).unsqueeze(0).to(seed_embs.device)
        seed_embs = seed_embs + pos_codes
        seed_embs = self.emb_dropout(seed_embs)

        # BatchSize x TargetLen x EmbSize
        target_features = seed_embs
        target_features = self.backbone(seed_embs,
                                        mask=dependency_mask,
                                        src_key_padding_mask=seed_padding_mask)
        logits = self.out(target_features)  # BatchSize x TargetLen x VocabSize
        return logits


def make_positional_encoding(max_length, embedding_size):
    time = np.pi * torch.arange(0, max_length).float()
    freq_dividers = torch.arange(1, embedding_size // 2 + 1).float()
    inputs = time[:, None] / freq_dividers[None, :]

    result = torch.zeros(max_length, embedding_size)
    result[:, 0::2] = torch.sin(inputs)
    result[:, 1::2] = torch.cos(inputs)
    return result


def make_target_dependency_mask(length):
    full_mask = torch.ones(length, length)
    ignore_mask = torch.tril(full_mask) < 1
    full_mask.masked_fill_(ignore_mask, float('-inf'))
    full_mask.masked_fill_(~ignore_mask, 0)
    return full_mask


torch_transf_model = LanguageModel(tokenizer.vocab_size(),
                                   256,
                                   BatchFirstTransformerEncoder(
                                       nn.TransformerEncoderLayer(
                                           d_model=256,
                                           nhead=16,
                                           dim_feedforward=512,
                                           dropout=0.1),
                                       num_layers=3),
                                   emb_dropout=0.1)

torch_transf_model.load_state_dict(
    torch.load("C:\\Users\\rober\\Downloads\\AI war_and_peace\\war_and_peace_torch_transf_best.pth",
               map_location=torch.device('cpu')))
greedy_generator = GreedyGenr(torch_transf_model, tokenizer, device='cpu')
class AI(GridLayout):
    def ai(self):
        texts = self.text1.text
        n = int(self.text2.text)
        self.lbl.text = greedy_generator(texts, n)


class aiApp(MDApp):
    tem_clc = ThemeManager()
    title = 'CESAR'

    def build(self):
        self.tem_clc.theme_style = 'Dark'
        return AI()


if __name__ == "__main__":
    aiApp().run()
