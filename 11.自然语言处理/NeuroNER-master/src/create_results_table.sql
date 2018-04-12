-- drop table results_neurodeid;

Create TABLE `results_neurodeid` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `dataset_name` varchar(100) DEFAULT NULL,
  `time_stamp` varchar(26) DEFAULT NULL,
  `train_f1_score` float DEFAULT NULL,
  `valid_f1_score` float DEFAULT NULL,
  `test_f1_score` float DEFAULT NULL,
  `test_f1_score_based_on_valid_` float DEFAULT NULL,
  `best_epoch_for_f1_score_based_on_valid_` float DEFAULT NULL,

    `train_f1_conll` float DEFAULT NULL,
  `valid_f1_conll` float DEFAULT NULL,
  `test_f1_conll` float DEFAULT NULL,
  `test_f1_conll_based_on_valid` float DEFAULT NULL,
  `best_epoch_for_f1_conll_based_on_valid_` float DEFAULT NULL,
  `test_f1_conll_based_on_valid_` float DEFAULT NULL,

  `char_dim` int(11) DEFAULT NULL,

`using_character_lstm`  int(11) DEFAULT NULL,
`char_lstm_dim`  int(11) DEFAULT NULL,
`char_bidirect`  int(11) DEFAULT NULL,

`using_character_cnn`  int(11) DEFAULT NULL,
`character_cnn_filter_height` int(11) DEFAULT NULL,
`character_cnn_number_of_filters` int(11) DEFAULT NULL,

`word_dim` int(11) DEFAULT NULL,
`using_token_lstm`  int(11) DEFAULT NULL,
`word_lstm_dim` int(11) DEFAULT NULL,
`word_bidirect` int(11) DEFAULT NULL,

`using_token_cnn` int(11) DEFAULT NULL,
`token_cnn_filter_height` int(11) DEFAULT NULL,
`token_cnn_number_of_filters` int(11) DEFAULT NULL,

`crf`  int(11) DEFAULT NULL,
`dropout` float DEFAULT NULL,
`lr_method` text DEFAULT NULL,
`training_set_size` float DEFAULT NULL,

  `all_emb` tinyint(1) DEFAULT NULL,
  `pre_emb` TEXT DEFAULT NULL,


  `dataset_text_filepath` varchar(300) DEFAULT NULL,
  `word_vector_filepath` varchar(300) DEFAULT NULL,
  `experiment_name` varchar(100) DEFAULT NULL,
  `patience` int(11) DEFAULT NULL,
  `best_epoch_for_f1_score_based_on_valid` int(11) DEFAULT NULL,
  `best_epoch_for_f1_score_monolabel_based_on_valid` int(11) DEFAULT NULL,
  `num_epochs` int(11) DEFAULT NULL,
  `early_stop` tinyint(1) DEFAULT NULL,
  `train_duration` float DEFAULT NULL,
  `keyboard_interrupt` tinyint(1) DEFAULT NULL,
  PRIMARY KEY (`id`)
)