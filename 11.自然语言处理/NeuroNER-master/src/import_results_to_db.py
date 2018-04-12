'''
Display overview of results

To install MySQLdb:
sudo apt-get install -y python-dev libmysqlclient-dev
sudo pip install MySQL-python

To run:
source ~/.bash_profile; cd /scratch/jjylee/neurodeid/src ; python import_results_to_db.py
'''
from __future__ import print_function
from __future__ import division

import os
import json
import utils
import numpy as np
import csv
import sys

def main():
    '''
    This is the main function
    '''
    #stats_graph_folder=os.path.join('..', 'stats_graphs', 'test')
    stats_graph_folder=os.path.join('..', 'output')

    # Getting a list of all subdirectories in the current directory. Not recursive.
    subfolders = os.listdir(stats_graph_folder)
    subfolders = sorted(os.listdir(stats_graph_folder), reverse=True)

    # Recursive
    #subfolders = [x[0] for x in os.walk(stats_graph_folder)][1:]

    # Parameters
    #metrics = ['accuracy_score', 'f1_score']
    metrics = ['f1_score','f1_conll']
    dataset_types = ['train', 'valid', 'test']
    execution_details = ['num_epochs', 'train_duration', 'keyboard_interrupt', 'early_stop']
    # It's good to put the important fields (for your experiments) first,
    # so that it appears right next to the test f1 score.

    fields_of_interest = '''dataset_text_filepath all_emb pre_emb char_dim  char_bidirect character_cnn_filter_height character_cnn_number_of_filters
    word_dim using_token_lstm word_lstm_dim word_bidirect experiment_name
    using_token_cnn token_cnn_filter_height token_cnn_number_of_filters using_character_lstm using_character_cnn patience  char_lstm_dim
    crf dropout lr_method training_set_size'''.replace('\n', '').split(' ')
    fields_of_interest = filter(None, fields_of_interest)

    result_tables = {}
    print('subfolders: {0}'.format(subfolders))
    # 0/0
    # Define column_order, i.e. how the result table is presented
    column_order = ['dataset_name', 'time_stamp']
    for metric in metrics:
        for dataset_type in dataset_types:
            column_order.append('{0}_{1}'.format(dataset_type, metric))
        column_order.append('{0}_{1} (based on valid)'.format('test', metric))

    column_order.extend(fields_of_interest[:3])

    for metric in metrics:
        column_order.append('best_epoch_for_{0} (based on valid)'.format(metric))
    column_order.extend(execution_details)

    column_order.extend(fields_of_interest[3:])

    print('fields_of_interest: {0}'.format(fields_of_interest))
    print('column_order: {0}'.format(column_order))


    for subfolder in subfolders:
        result_row = {}
        result_filepath = os.path.join(stats_graph_folder, subfolder, 'results.json')
        if not os.path.isfile(result_filepath): continue
        print('result_filepath: {0}'.format(result_filepath))
        try:
            result_json = json.load(open(result_filepath, 'r'))
        except ValueError:
            print('This file is skipped since it is in use or corrupted.')

        # Include time stamp of the experiments
        result_row['time_stamp'] =  result_json['execution_details']['time_stamp']

        for field_of_interest in fields_of_interest:
            if field_of_interest in result_json['model_options']:
                if field_of_interest == 'pre_emb':
                    result_row[field_of_interest] = os.path.basename(result_json['model_options'][field_of_interest])
                else:
                    result_row[field_of_interest] = result_json['model_options'][field_of_interest]

        for execution_detail in execution_details:
            try:
                result_row[execution_detail] = result_json['execution_details'][execution_detail]
            except:
                result_row[execution_detail] = 'NULL'

        for metric in metrics:
            for dataset_type in dataset_types:
                    result_row['{0}_{1}'.format(dataset_type, metric)] = result_json[dataset_type].get('best_{0}'.format(metric), 'NULL')
                    if dataset_type == 'test':
                        result_row['{0}_{1} (based on valid)'.format(dataset_type, metric)] = result_json[dataset_type].get('best_{0}_based_on_valid'.format(metric), 'NULL')
                    elif dataset_type == 'valid':
                        result_row['best_epoch_for_{0} (based on valid)'.format(metric)] = result_json[dataset_type].get('epoch_for_best_{0}'.format(metric), 'NULL')

        # Save row in table: one table per data set
        dataset_name = utils.get_basename_without_extension(result_row['dataset_text_filepath'])
        result_row['dataset_name'] = dataset_name
        if dataset_name not in result_tables:
            result_tables[dataset_name] = []



        result_row_ordered = []
        for column_name in column_order:
            if column_name in result_row:
                result_row_ordered.append(result_row[column_name])
            else:
                result_row_ordered.append('NULL')


        result_tables[dataset_name].append(result_row_ordered)

    print('result_tables: {0}'.format(result_tables))

    #print('\ncolumn_order: {0}'.format(column_order))
    #print('result_table: {0}'.format(result_tables))


    import MySQLdb as mdb
    connection = mdb.connect('128.52.165.241', 'tc', open('database_password.txt', 'r').readline(), 'tc');
    cursor = connection.cursor()

    for dataset_name, dataset_results in result_tables.items():
        with open(os.path.join(stats_graph_folder, 'results_{0}.csv'.format(dataset_name)), 'wb') as testfile:
            csv_writer = csv.writer(testfile)
            clean_column_names = map(clean_column_name, column_order)
            csv_writer.writerow(clean_column_names)
            for row in dataset_results:
                csv_writer.writerow(row)

                # Convert row values to some importable string
                values = ''
                for value_number, value in enumerate(row):
                    if value_number > 0:
                        values += ','
                    if isinstance(value, (bool)): # Check if object is a boolean
                        if value:
                            value = '1'
                        else:
                            value = '0'
                    if not isinstance( value, ( int, long ) ) and value != 'NULL': # http://stackoverflow.com/questions/3501382/checking-whether-a-variable-is-an-integer-or-not
                        value = '"{0}"'.format(str(value))
                    else:
                        value = '{0}'.format(str(value))
                    #print('value: {0}'.format(value))
                    values += value

                # Make sure that train_duration is more than 0. (if 0 it means the training was interrupted)
                #print('row[clean_column_names.index("train_duration"): {0}'.format(row[clean_column_names.index('train_duration')]))
                train_duration = row[clean_column_names.index('train_duration')]
                keyboard_interrupt = row[clean_column_names.index('keyboard_interrupt')]
                #print('train_duration: {0}'.format(train_duration))
                '''
                if train_duration == 'NULL' and keyboard_interrupt == '0' or train_duration == '0':
                    print('The experiment has train_duration = {0}, so we skip it.'.format(train_duration))
                    continue
                '''

                # Make sure the experiment isn't already in the database
                time_stamp = row[clean_column_names.index('time_stamp')]
                sql = 'SELECT COUNT(*) FROM tc.results_neurodeid WHERE time_stamp = "{0}"'.format(time_stamp)
                cursor.execute(sql)
                row = cursor.fetchone()
                if row[0]  >= 1:
                    print('The experiment with timestamp {0} is already in the database, so we skip it.'.format(time_stamp))
                    continue
                if time_stamp < '2016-08-17_18-20-05-836274':
                    print('The experiment with timestamp {0} is too old, so we skip it.'.format(time_stamp))
                    continue


                sql = 'INSERT INTO tc.results_neurodeid ({0}) VALUES ({1})'.format(','.join(clean_column_names), values)

                print('sql: {0}'.format(sql))
                cursor.execute(sql)
                connection.commit()


    connection.commit()
    connection.close()

def clean_column_name(column_name):
    return column_name.replace(' ', '_').replace(')', '_').replace('(', '')




if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling