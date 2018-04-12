-- SELECT 
--   id, dataset_name, time_stamp, train_f1_score, valid_f1_score, test_f1_score, test_f1_score_based_on_valid_, best_epoch_for_f1_score_based_on_valid_,    train_f1_conll, valid_f1_conll, test_f1_conll, test_f1_conll_based_on_valid, best_epoch_for_f1_conll_based_on_valid_, test_f1_conll_based_on_valid_,  char_dim,
-- using_character_lstm, char_lstm_dim, char_bidirect, using_character_cnn, character_cnn_filter_height, character_cnn_number_of_filters, word_dim, using_token_lstm, word_lstm_dim, word_bidirect, using_token_cnn, token_cnn_filter_height, token_cnn_number_of_filters, crf, dropout, lr_method, training_set_size,     all_emb, pre_emb,
--   dataset_text_filepath, word_vector_filepath, experiment_name, patience, best_epoch_for_f1_score_based_on_valid, best_epoch_for_f1_score_monolabel_based_on_valid, num_epochs, early_stop, train_duration, keyboard_interrupt FROM tc.results_neurodeid
SELECT 
  id, dataset_name, time_stamp, train_f1_score, valid_f1_score, test_f1_score, test_f1_score_based_on_valid_ best_test_f1, best_epoch_for_f1_score_based_on_valid_ best_epoch,    train_f1_conll, valid_f1_conll, test_f1_conll, test_f1_conll_based_on_valid_ best_test_f1_conll, best_epoch_for_f1_conll_based_on_valid_ conll_epoch,  num_epochs,   
using_character_lstm, crf, dropout, pre_emb, word_vector_filepath, experiment_name, early_stop, train_duration, keyboard_interrupt FROM tc.results_neurodeid
WHERE char_dim=25 AND char_lstm_dim=25 AND word_dim=100 AND dropout=0.5 AND dataset_name='en'
ORDER BY time_stamp DESC, using_character_lstm,crf, best_test_f1_conll DESC;

