post_file('send_log.m', 'http://127.0.0.1:5000/upload_file');
fetch_file_name('http://127.0.0.1:5000/example_data')
send_log('10.2', 'Fault','http://127.0.0.1:5000/log_event');
