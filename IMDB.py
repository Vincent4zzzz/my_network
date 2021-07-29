import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers,losses

batch_size = 128
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)
word_index = keras.datasets.imdb.get_word_index()
for k, v in word_index.items():
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0  # 填充标志
    word_index["<START>"] = 1  # 起始标志
    word_index["<UNK>"] = 2  # 未知单词的标志
    word_index["<UNUSED>"] = 3
#reverse_word_index = {[[value, key] for [key, value] in word_index.items()]}
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 截断和填充句子，使所有句子等长
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.shuffle.batch(batch_size, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class my_RNN(keras.Model):
    def __init__(self,units):
        super(my_RNN,self).__init__()
        self.state0=[tf.zeros([batch_size,units])]
        self.state1=[tf.zeros(batch_size,units)]
        self.embedding=layers.embedding(total_words,embedding_len,input_length=max_review_len)
        self.rnn_cell0=layers.SimpleRNNCell(units,dropout=0.5)
        self.rnn_cell1=layers.SimpleRNNCell(units,dropout=0.5)
        self.outlayer=layers.Dense(1)
    def call(self,inputs,training=None):
        x=inputs
        x=self.embedding(x)
        state0=self.state0
        state1=self.state1
        for word in tf.unstack(x,axis=1):
            out0,state0=self.rnn_cell0(word,state0,training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        x=self.outlayer(out1,training)
        prob=tf.sigmoid(x)
        return prob
def main():
    units=64
    epoch=80
    model=my_RNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=losses.CategoricalCrossentropy(),
                  metrics=['Accuracy'])
    model.fit(db_train,epoch=epoch,validation_data=db_test)
    model.evaluate(db_test)

if "__main__":
    main()
