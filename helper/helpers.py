import re


def normalize(message):
    if type(message) is not str:
        return ''
    message = re.sub('[^a-zA-Za-яA-Я1-9]', ' ', message)

    # Emoticones y Pictogramas (U+1F600 - U+1F64F)
    # Símbolos y Pictogramas (U+1F300 - U+1F5FF)
    # Banderas de países (U+1F1E0 - U+1F1FF)
    # Símbolos varios (U+02702 - U+027B0)
    # Símbolos de comunicación (U+024C2 - U+1F251)
    message = re.sub(
        '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]',
        '', message)
    message = re.sub(' +', ' ', message)

    return message.strip()
