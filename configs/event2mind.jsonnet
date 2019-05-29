{
  "dataset_reader": {
    "type": "event2mind_dataset_reader",
    // Uncomment this when generating the vocabularly with `dry-run`.
    //"dummy_instances_for_vocab_generation": true,
    "source_tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "spacy"
      }
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    }
  },
  "vocabulary": {
    // Uncomment this when generating the vocabularly with `dry-run`.
    //"min_count": {"source_tokens": 2}
    // Uncomment this when training using an existing vocabularly.
    //"directory_path": "output_dir/vocabulary/"
  },
  "train_data_path": "docs/data/train.csv",
  "validation_data_path": "docs/data/dev.csv",
  "model": {
    "type": "event2mind_model",
    "source_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "source_tokens",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz",
          "embedding_dim": 300,
          "trainable": false
        }
      }
    },
    "embedding_dropout": 0.2,
    "encoder": {
      "type": "gru",
      "input_size": 300,
      // When we concatenate the forward and backward states together this gives
      // our desired encoded vector of size 100.
      "hidden_size": 50,
      "num_layers": 1,
      "bidirectional": true
    },
    "max_decoding_steps": 10,
    // Following the original model we use a single namespace.
    "target_namespace": "source_tokens"
  },
  "iterator": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size" : 64,
    "sorting_keys": [["source", "num_tokens"]]
  },
  "trainer": {
    "num_epochs": 10,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam"
    },
    "validation_metric": "+xintent"
  }
}