import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import plotly.express as px
import plotly.io as pio
import kaleido

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300
HIDDEN_LSTM_DIM = 100  # todo add this line to the README
DROP_OUT = 0.5
LSTM_LAYERS = 1

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
RARE = "rare"
NEGATED_POLARITY = "negated_polarity"

# todo: we made this lines so add to README§
TRAIN_LOSSES, TRAIN_ACCURACIES = [], []
VAL_LOSSES, VAL_ACCURACIES = [], []


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# todo: we made this function add to README
def plot(graph_title, axis_names, curves, curves_titles):
    fig = px.line(x=list(range(1, len(curves[0]) + 1)), y=curves,
                  title=graph_title, )

    fig.update_layout(xaxis_title=axis_names[0], yaxis_title=axis_names[1],
                      xaxis=dict(tickvals=list(range(1, len(curves[0]) + 1)), tickmode='array'))

    for i, title in enumerate(curves_titles):
        fig.update_traces(name=title, selector=dict(name=f'wide_variable_{i}'))

    pio.write_image(fig, f'{graph_title}.png')
    fig.show()


# todo add this function to the README
def append_to_file(filename, text):
    """
    Appends the given text to the bottom of the specified file.

    Args:
        filename: The name of the text file.
        text: The string to be appended.
    """
    with open(filename, "a") as file:
        file.write(text + "\n")


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    # Initialize an empty numpy array with zeros
    avg_vector = np.zeros(embedding_dim)
    # Iterate over each word in the sentence
    for word in sent.text:
        # Check if the word is in the word_to_vec dictionary
        if word in word_to_vec:
            # Add the word embedding to the running total
            avg_vector += word_to_vec[word]
    # Divide the total by the number of words in the sentence
    if len(sent.text) != 0:
        avg_vector /= len(sent.text)
    # Return the average word vector
    return avg_vector


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot_embedding = np.zeros(size)
    one_hot_embedding[ind] = 1
    return one_hot_embedding


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)
    one_hot = np.zeros(size)
    for word in sent.text:
        new_one_hot = np.zeros(size)
        new_one_hot[word_to_ind[word]] = 1
        one_hot += new_one_hot
    return one_hot / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {word: i for i, word in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    # Initialize an empty numpy ndarray with zeros size of 52x300 = (seq_len, embedding_dim)
    seq_vector = np.zeros((seq_len, embedding_dim))

    # Iterate over each word in the sentence
    for i in range(min(len(sent.text), seq_len)):
        # Check if the word is in the word_to_vec mapping
        if sent.text[i] in word_to_vec:
            # Add the word embedding to the running total
            seq_vector[i] = word_to_vec[sent.text[i]]

    # Return the average word vector
    return seq_vector


# todo add this function to the README
def get_rare_words_sentences(sentences, dataset):
    """
    :param sentences: list of sentences from the test set
    :param dataset: the SentimentTreeBank object
    :return: list of all the sentences indices which are considered as rare words sentences
    """
    rare_words_indices = data_loader.get_rare_words_examples(sentences, dataset)
    rare_words_sentences = []
    for i in rare_words_indices:
        rare_words_sentences.append(sentences[i])
    return rare_words_sentences


# todo add this function to the README
def get_negated_polarity_sentences(sentences):
    """
    :param sentences: list of sentences from the test set
    :return: list of all the sentences indices which are considered as negated polarity sentences
    """
    negated_polarity_indices = data_loader.get_negated_polarity_examples(sentences)
    negated_polarity_sentences = []
    for i in negated_polarity_indices:
        negated_polarity_sentences.append(sentences[i])
    return negated_polarity_sentences


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()
        self.sentences[RARE] = get_rare_words_sentences(self.sentences[TEST], self.sentiment_dataset)
        self.sentences[NEGATED_POLARITY] = get_negated_polarity_sentences(self.sentences[TEST])

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout,
                            bidirectional=True, device=get_available_device(), batch_first=True, dtype=torch.float64)
        self.linear = nn.Linear(in_features=hidden_dim * 2, out_features=1, bias=True, dtype=torch.float64)
        return

    def forward(self, text):
        lstm_output, (hn, cn) = self.lstm(text)
        res = self.linear(torch.hstack([hn[0], hn[1]]))
        return res.squeeze()

    def predict(self, text):
        return torch.round(nn.Sigmoid()(self.forward(text)))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linearLayer = nn.Linear(in_features=embedding_dim, out_features=1, dtype=torch.float64, bias=True,
                                     device=get_available_device())

    def forward(self, x):
        return self.linearLayer(x).squeeze()

    def predict(self, x):
        return torch.round(nn.Sigmoid()(self.linearLayer(x).squeeze()))


# ------------------------- training functions -------------

# todo add to README that not used
def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return (preds == y).sum() / len(y)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    if len(data_iterator) == 0:
        raise Exception("Data iterator is empty!")

    model.requires_grad_(True)

    for x, y in data_iterator:
        optimizer.zero_grad()
        y_pred = model(x.to(get_available_device()))
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    return evaluate(model, data_iterator, criterion)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    if len(data_iterator) == 0:
        raise Exception("Data iterator is empty!")

    model.requires_grad_(False)
    acc, loss, sample_counters = 0, 0, 0

    for x, y in data_iterator:
        y_pred = model(x.to(get_available_device()))
        loss += criterion(y_pred, y).item()
        sample_counters += len(y)
        acc += ((torch.round(nn.Sigmoid()(y_pred)) == y).sum()).item()

    return np.round(loss / sample_counters, 5), np.round(100 * acc / sample_counters, 5)


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    model.requires_grad_(False)
    predictions = torch.Tensor()

    for x in data_iter:
        y_pred = model.predict(x)
        torch.cat((predictions, y_pred))

    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(device=get_available_device())
    train_data_iterator = data_manager.get_torch_iterator(TRAIN)
    val_data_iterator = data_manager.get_torch_iterator(VAL)

    TRAIN_LOSSES.clear()
    TRAIN_ACCURACIES.clear()
    VAL_LOSSES.clear()
    VAL_ACCURACIES.clear()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_data_iterator, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_data_iterator, criterion)
        TRAIN_LOSSES.append(train_loss)
        TRAIN_ACCURACIES.append(train_acc)
        VAL_LOSSES.append(val_loss)
        VAL_ACCURACIES.append(val_acc)
    return


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64, use_sub_phrases=True)
    log_linear = LogLinear(data_manager.get_input_shape()[0])
    train_model(log_linear, data_manager, n_epochs=20, lr=0.01, weight_decay=0.001)
    return log_linear, data_manager  # todo add this api changes to the readme


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    # model = LogLinear(data_manager.get_input_shape()[0]).to(get_available_device())
    log_linear_w2v = LogLinear(embedding_dim=W2V_EMBEDDING_DIM)
    train_model(log_linear_w2v, data_manager, n_epochs=20, lr=0.01, weight_decay=0.001)
    return log_linear_w2v, data_manager  # todo add this line change to readme


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    lstm_w2v = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=HIDDEN_LSTM_DIM, n_layers=LSTM_LAYERS, dropout=DROP_OUT)
    train_model(lstm_w2v, data_manager, n_epochs=4, lr=0.001, weight_decay=0.0001)
    return lstm_w2v, data_manager  # todo add this line change to readme


# TODO add this function to the README
def answer(train_model_function, title, lr, weight_decay, n_epochs):
    print("start training", title)
    model, data = train_model_function()

    plot(f"{title} Train & Validation Losses", ["Epoch number", "Loss"],
         [TRAIN_LOSSES, VAL_LOSSES], ["Train", "Validation"])
    plot(f"{title} Train & Validation Accuracies", ["Epoch number", "Accuracy"],
         [TRAIN_ACCURACIES, VAL_ACCURACIES], ["Train", "Validation"])

    # test
    test_loss, test_acc = evaluate(model, data.get_torch_iterator(TEST), nn.BCEWithLogitsLoss(reduction='sum'))
    print(f"{title} Test Evaluation:")
    print(f"Test Loss: {test_loss} | Test Acc: {test_acc}%\n")

    # Negated Polarity
    negated_polarity_loss, negated_polarity_acc = evaluate(model, data.get_torch_iterator(NEGATED_POLARITY),
                                                           nn.BCEWithLogitsLoss(reduction='sum'))
    negated_polarity_res = (f"{title} Negated Polarity Evaluation:\n"
                            f" Negated Polarity Loss: {negated_polarity_loss} | "
                            f"Negated Polarity Acc: {negated_polarity_acc}%\n")
    print(negated_polarity_res)

    # Rare Words
    rare_loss, rare_acc = evaluate(model, data.get_torch_iterator(RARE), nn.BCEWithLogitsLoss(reduction='sum'))
    rare_res = f"{title} Rare Words Evaluation:\n Rare Loss: {rare_loss} | Rare Acc: {rare_acc}%\n"
    print(rare_res)

    append_to_file("results.txt", "Latest results:\n" + rare_res + negated_polarity_res + "\n")

    save_model(model, f"{title}_model", n_epochs,
               optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay))
    print(f"{title} model saved as {title}_model")
    print("**********************\n\n")


if __name__ == '__main__':
    answer(train_log_linear_with_one_hot, "LogLinear one hot", lr=0.01, weight_decay=0.001, n_epochs=20)
    answer(train_log_linear_with_w2v, "LogLinear w2v", lr=0.01, weight_decay=0.001, n_epochs=20)
    answer(train_lstm_with_w2v, "LSTM w2v", lr=0.001, weight_decay=0.0001, n_epochs=4)
