import torch



def get_sentiment(sentence, tokenizer, model):

    tokens = tokenizer.encode(sentence, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

def get_number_of_unique_words(comments):
    unique_words = set()
    for sentence in comments:
        for word in sentence.split(" "):
            unique_words = unique_words.union({word})
            
    return len(unique_words)