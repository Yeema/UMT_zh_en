from googletrans import Translator
translator = Translator()

def self_trans(text):
    res = translator.translate(text)
    return res.text
